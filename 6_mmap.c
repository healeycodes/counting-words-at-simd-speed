#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <arm_neon.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>

#define PREFETCH_DISTANCE 256  // Prefetch ahead for cache optimization

// Optimized whitespace check using lookup table
static const uint8_t ws_lut[256] = {
    [' '] = 1, ['\n'] = 1, ['\r'] = 1, 
    ['\t'] = 1, ['\v'] = 1, ['\f'] = 1
};

static inline bool is_ws_scalar(unsigned char c)
{
    return ws_lut[c];
}

// Process buffer with unrolled NEON - process 64 bytes per iteration
static size_t process_buffer_neon_unrolled(const unsigned char *buffer, size_t len, 
                                           uint8x16_t *prev_ws_vec, bool *prev_ws)
{
    size_t words = 0;
    size_t i = 0;

    // Whitespace comparison vectors - precomputed
    const uint8x16_t space_vec = vdupq_n_u8(' ');
    const uint8x16_t newline_vec = vdupq_n_u8('\n');
    const uint8x16_t cr_vec = vdupq_n_u8('\r');
    const uint8x16_t tab_vec = vdupq_n_u8('\t');
    const uint8x16_t vtab_vec = vdupq_n_u8('\v');
    const uint8x16_t ff_vec = vdupq_n_u8('\f');

    // Process 64 bytes at a time (4x unrolled)
    if (len >= 64)
    {
        size_t nvec = len & ~(size_t)63;
        for (; i < nvec; i += 64)
        {
            // Prefetch next cache lines
            __builtin_prefetch(buffer + i + PREFETCH_DISTANCE, 0, 3);
            
            // Load 64 bytes (4x16)
            uint8x16_t bytes0 = vld1q_u8(buffer + i);
            uint8x16_t bytes1 = vld1q_u8(buffer + i + 16);
            uint8x16_t bytes2 = vld1q_u8(buffer + i + 32);
            uint8x16_t bytes3 = vld1q_u8(buffer + i + 48);

            // Compute whitespace masks for all 4 vectors in parallel
            uint8x16_t ws0 = vceqq_u8(bytes0, space_vec);
            uint8x16_t ws1 = vceqq_u8(bytes1, space_vec);
            uint8x16_t ws2 = vceqq_u8(bytes2, space_vec);
            uint8x16_t ws3 = vceqq_u8(bytes3, space_vec);
            
            ws0 = vorrq_u8(ws0, vceqq_u8(bytes0, newline_vec));
            ws1 = vorrq_u8(ws1, vceqq_u8(bytes1, newline_vec));
            ws2 = vorrq_u8(ws2, vceqq_u8(bytes2, newline_vec));
            ws3 = vorrq_u8(ws3, vceqq_u8(bytes3, newline_vec));
            
            ws0 = vorrq_u8(ws0, vceqq_u8(bytes0, cr_vec));
            ws1 = vorrq_u8(ws1, vceqq_u8(bytes1, cr_vec));
            ws2 = vorrq_u8(ws2, vceqq_u8(bytes2, cr_vec));
            ws3 = vorrq_u8(ws3, vceqq_u8(bytes3, cr_vec));
            
            ws0 = vorrq_u8(ws0, vceqq_u8(bytes0, tab_vec));
            ws1 = vorrq_u8(ws1, vceqq_u8(bytes1, tab_vec));
            ws2 = vorrq_u8(ws2, vceqq_u8(bytes2, tab_vec));
            ws3 = vorrq_u8(ws3, vceqq_u8(bytes3, tab_vec));
            
            ws0 = vorrq_u8(ws0, vceqq_u8(bytes0, vtab_vec));
            ws1 = vorrq_u8(ws1, vceqq_u8(bytes1, vtab_vec));
            ws2 = vorrq_u8(ws2, vceqq_u8(bytes2, vtab_vec));
            ws3 = vorrq_u8(ws3, vceqq_u8(bytes3, vtab_vec));
            
            ws0 = vorrq_u8(ws0, vceqq_u8(bytes0, ff_vec));
            ws1 = vorrq_u8(ws1, vceqq_u8(bytes1, ff_vec));
            ws2 = vorrq_u8(ws2, vceqq_u8(bytes2, ff_vec));
            ws3 = vorrq_u8(ws3, vceqq_u8(bytes3, ff_vec));

            // Process first vector
            uint8x16_t prev_ws_shifted0 = vextq_u8(*prev_ws_vec, ws0, 15);
            uint8x16_t non_ws0 = vmvnq_u8(ws0);
            uint8x16_t start_mask0 = vandq_u8(non_ws0, prev_ws_shifted0);
            words += vaddvq_u8(vshrq_n_u8(start_mask0, 7));

            // Process second vector
            uint8x16_t prev_ws_shifted1 = vextq_u8(ws0, ws1, 15);
            uint8x16_t non_ws1 = vmvnq_u8(ws1);
            uint8x16_t start_mask1 = vandq_u8(non_ws1, prev_ws_shifted1);
            words += vaddvq_u8(vshrq_n_u8(start_mask1, 7));

            // Process third vector
            uint8x16_t prev_ws_shifted2 = vextq_u8(ws1, ws2, 15);
            uint8x16_t non_ws2 = vmvnq_u8(ws2);
            uint8x16_t start_mask2 = vandq_u8(non_ws2, prev_ws_shifted2);
            words += vaddvq_u8(vshrq_n_u8(start_mask2, 7));

            // Process fourth vector
            uint8x16_t prev_ws_shifted3 = vextq_u8(ws2, ws3, 15);
            uint8x16_t non_ws3 = vmvnq_u8(ws3);
            uint8x16_t start_mask3 = vandq_u8(non_ws3, prev_ws_shifted3);
            words += vaddvq_u8(vshrq_n_u8(start_mask3, 7));

            // Update state for next iteration
            *prev_ws_vec = ws3;
        }

        if (i > 0)
        {
            *prev_ws = is_ws_scalar(buffer[i - 1]);
        }
    }

    // Process remaining 16-byte chunks
    if (i + 16 <= len)
    {
        size_t nvec = len & ~(size_t)15;
        for (; i < nvec; i += 16)
        {
            uint8x16_t bytes = vld1q_u8(buffer + i);

            uint8x16_t ws = vceqq_u8(bytes, space_vec);
            ws = vorrq_u8(ws, vceqq_u8(bytes, newline_vec));
            ws = vorrq_u8(ws, vceqq_u8(bytes, cr_vec));
            ws = vorrq_u8(ws, vceqq_u8(bytes, tab_vec));
            ws = vorrq_u8(ws, vceqq_u8(bytes, vtab_vec));
            ws = vorrq_u8(ws, vceqq_u8(bytes, ff_vec));

            uint8x16_t prev_ws_shifted = vextq_u8(*prev_ws_vec, ws, 15);
            uint8x16_t non_ws = vmvnq_u8(ws);
            uint8x16_t start_mask = vandq_u8(non_ws, prev_ws_shifted);
            words += vaddvq_u8(vshrq_n_u8(start_mask, 7));

            *prev_ws_vec = ws;
        }

        if (i > 0)
        {
            *prev_ws = is_ws_scalar(buffer[i - 1]);
        }
    }

    // Process remaining bytes with unrolled scalar loop
    size_t remaining = len - i;
    if (remaining >= 8)
    {
        size_t unroll_end = i + (remaining & ~(size_t)7);
        for (; i < unroll_end; i += 8)
        {
            // Process 8 bytes at a time
            bool cur_ws0 = is_ws_scalar(buffer[i]);
            bool cur_ws1 = is_ws_scalar(buffer[i + 1]);
            bool cur_ws2 = is_ws_scalar(buffer[i + 2]);
            bool cur_ws3 = is_ws_scalar(buffer[i + 3]);
            bool cur_ws4 = is_ws_scalar(buffer[i + 4]);
            bool cur_ws5 = is_ws_scalar(buffer[i + 5]);
            bool cur_ws6 = is_ws_scalar(buffer[i + 6]);
            bool cur_ws7 = is_ws_scalar(buffer[i + 7]);
            
            words += (!cur_ws0 && *prev_ws);
            words += (!cur_ws1 && cur_ws0);
            words += (!cur_ws2 && cur_ws1);
            words += (!cur_ws3 && cur_ws2);
            words += (!cur_ws4 && cur_ws3);
            words += (!cur_ws5 && cur_ws4);
            words += (!cur_ws6 && cur_ws5);
            words += (!cur_ws7 && cur_ws6);
            
            *prev_ws = cur_ws7;
        }
    }

    // Handle final bytes
    for (; i < len; ++i)
    {
        bool cur_ws = is_ws_scalar(buffer[i]);
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

    int fd = open(argv[1], O_RDONLY);
    if (fd < 0)
    {
        perror("open");
        return 1;
    }

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

    // Memory map the file
    unsigned char *data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED)
    {
        perror("mmap");
        close(fd);
        return 1;
    }

    // Advise kernel about access pattern
    // MADV_SEQUENTIAL: We're reading sequentially
    // MADV_WILLNEED: Pre-fault pages into memory
    madvise(data, st.st_size, MADV_SEQUENTIAL | MADV_WILLNEED);

    // Process the entire file
    bool prev_ws = true;
    uint8x16_t prev_ws_vec = vdupq_n_u8(0xFF);
    size_t words = process_buffer_neon_unrolled(data, st.st_size, &prev_ws_vec, &prev_ws);

    munmap(data, st.st_size);
    close(fd);

    printf("%zu\n", words);
    return 0;
}
