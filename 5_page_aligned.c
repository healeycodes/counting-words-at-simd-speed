#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <arm_neon.h>
#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <string.h>

#define PAGE_SIZE 8192

typedef struct {
    size_t total_words;
    bool prev_ws;
    uint8x16_t prev_ws_vec;
    dispatch_queue_t process_queue;
} shared_state_t;

static inline bool is_ws_scalar(unsigned char c)
{
    switch (c)
    {
    case ' ':
    case '\n':
    case '\r':
    case '\t':
    case '\v':
    case '\f':
        return true;
    default:
        return false;
    }
}

static size_t process_buffer_neon(const unsigned char *buffer, size_t len, uint8x16_t *prev_ws_vec, bool *prev_ws)
{
    size_t words = 0;
    size_t i = 0;

    if (len >= 16)
    {
        size_t nvec = len & ~(size_t)15;
        for (; i < nvec; i += 16)
        {
            uint8x16_t bytes = vld1q_u8(buffer + i);

            uint8x16_t ws1 = vceqq_u8(bytes, vdupq_n_u8(' '));
            uint8x16_t ws2 = vceqq_u8(bytes, vdupq_n_u8('\n'));
            uint8x16_t ws3 = vceqq_u8(bytes, vdupq_n_u8('\r'));
            uint8x16_t ws4 = vceqq_u8(bytes, vdupq_n_u8('\t'));
            uint8x16_t ws5 = vceqq_u8(bytes, vdupq_n_u8('\v'));
            uint8x16_t ws6 = vceqq_u8(bytes, vdupq_n_u8('\f'));

            uint8x16_t ws = vorrq_u8(ws1, ws2);
            ws = vorrq_u8(ws, ws3);
            ws = vorrq_u8(ws, ws4);
            ws = vorrq_u8(ws, ws5);
            ws = vorrq_u8(ws, ws6);

            uint8x16_t prev_ws_shifted = vextq_u8(*prev_ws_vec, ws, 15);

            uint8x16_t non_ws = vmvnq_u8(ws);
            uint8x16_t start_mask = vandq_u8(non_ws, prev_ws_shifted);

            uint8x16_t ones = vshrq_n_u8(start_mask, 7);

            words += (size_t)vaddvq_u8(ones);

            *prev_ws_vec = ws;
        }

        if (i > 0)
        {
            *prev_ws = is_ws_scalar(buffer[i - 1]);
        }
    }

    for (; i < len; ++i)
    {
        unsigned char c = buffer[i];
        bool cur_ws = is_ws_scalar(c);
        if (!cur_ws && *prev_ws)
        {
            ++words;
        }
        *prev_ws = cur_ws;
    }

    return words;
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "usage: wc <file>\n");
        return 1;
    }

    // Open file with O_RDONLY for dispatch_io
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0)
    {
        perror("open");
        return 1;
    }

    // Get file size for optimization
    struct stat st;
    if (fstat(fd, &st) < 0)
    {
        perror("fstat");
        close(fd);
        return 1;
    }

    if (st.st_size == 0)
    {
        close(fd);
        printf("0\n");
        return 0;
    }

    // Initialize state
    shared_state_t *state = calloc(1, sizeof(shared_state_t));
    if (!state)
    {
        perror("calloc");
        close(fd);
        return 1;
    }
    
    state->prev_ws = true;
    state->prev_ws_vec = vdupq_n_u8(0xFF);
    
    // Create a serial queue for processing to maintain order
    state->process_queue = dispatch_queue_create("process.queue", DISPATCH_QUEUE_SERIAL);

    // Create dispatch queue for I/O
    dispatch_queue_t io_queue = dispatch_queue_create("io.queue", DISPATCH_QUEUE_CONCURRENT);
    dispatch_group_t group = dispatch_group_create();

    // Create dispatch I/O channel
    dispatch_io_t channel = dispatch_io_create(DISPATCH_IO_STREAM, fd,
                                                io_queue, ^(int error) {
        if (error)
        {
            fprintf(stderr, "dispatch_io error: %d\n", error);
        }
        close(fd);
    });

    // Set the low water mark for reading
    dispatch_io_set_low_water(channel, PAGE_SIZE);

    // Read the entire file with dispatch_io_read
    // Using off_t 0 and size_t SIZE_MAX to read the entire file
    // dispatch_io will automatically chunk it into pages
    dispatch_group_enter(group);
    
    dispatch_io_read(channel, 0, SIZE_MAX, io_queue,
                     ^(bool done, dispatch_data_t data, int error) {
        if (error)
        {
            fprintf(stderr, "Read error: %d\n", error);
            if (done)
            {
                dispatch_group_leave(group);
            }
            return;
        }

        if (data && dispatch_data_get_size(data) > 0)
        {
            size_t data_size = dispatch_data_get_size(data);
            
            // Allocate buffer for this chunk
            unsigned char *buffer = (unsigned char *)aligned_alloc(16, data_size);
            if (!buffer)
            {
                fprintf(stderr, "Failed to allocate buffer\n");
                if (done)
                {
                    dispatch_group_leave(group);
                }
                return;
            }
            
            // Copy data to buffer
            __block size_t copied = 0;
            dispatch_data_apply(data, ^bool(__unused dispatch_data_t region,
                                            __unused size_t offset_in_region,
                                            const void *bytes,
                                            size_t size) {
                memcpy(buffer + copied, bytes, size);
                copied += size;
                return true;
            });
            
            // Process this chunk on the serial processing queue
            // This ensures chunks are processed in order
            dispatch_async(state->process_queue, ^{
                size_t words = process_buffer_neon(buffer, data_size, 
                                                   &state->prev_ws_vec, 
                                                   &state->prev_ws);
                state->total_words += words;
                free(buffer);
            });
        }

        if (done)
        {
            dispatch_group_leave(group);
        }
    });

    // Wait for all I/O and processing to complete
    dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
    
    // Ensure all processing is done
    dispatch_sync(state->process_queue, ^{
        // This block ensures all previous async processing is complete
    });

    // Cleanup
    dispatch_io_close(channel, 0);
    dispatch_release(channel);
    dispatch_release(io_queue);
    dispatch_release(state->process_queue);
    dispatch_release(group);

    printf("%zu\n", state->total_words);
    
    free(state);
    return 0;
}
