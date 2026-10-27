#include "tensor_loading.h"

#include <stdio.h>
#include <stdlib.h>

float *load_f32_file(const char *path, size_t expected_count) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "failed to open %s\n", path);
        return NULL;
    }

    if (fseek(file, 0, SEEK_END) != 0) {
        fprintf(stderr, "failed to seek %s\n", path);
        fclose(file);
        return NULL;
    }

    long file_size = ftell(file);
    if (file_size < 0) {
        fprintf(stderr, "failed to read size for %s\n", path);
        fclose(file);
        return NULL;
    }
    rewind(file);

    size_t expected_bytes = expected_count * sizeof(float);
    if ((size_t)file_size != expected_bytes) {
        fprintf(stderr, "unexpected size for %s: expected %zu bytes, got %ld bytes\n", path,
                expected_bytes, file_size);
        fclose(file);
        return NULL;
    }

    float *values = malloc(expected_bytes);
    if (values == NULL) {
        fprintf(stderr, "failed to allocate %zu bytes for %s\n", expected_bytes, path);
        fclose(file);
        return NULL;
    }

    size_t values_read = fread(values, sizeof(float), expected_count, file);
    fclose(file);

    if (values_read != expected_count) {
        fprintf(stderr, "failed to read %s: expected %zu floats, got %zu\n", path, expected_count,
                values_read);
        free(values);
        return NULL;
    }

    return values;
}
