void output(void) {
    // ruleid: c-no-stdout
    printf("hello\n");
    // ruleid: c-no-stdout
    puts("hello");
    // ok: c-no-stdout
    fprintf(stderr, "error\n");
    // ok: c-no-stdout
    snprintf(buffer, sizeof(buffer), "%d", 1);
    // ok: c-no-stdout
    // printf("in a comment\n");
}
