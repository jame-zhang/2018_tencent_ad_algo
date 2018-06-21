//
// Created by iioiio on 18-6-5.
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

// 查找参数
int find_arg(char *str, int argc, char **argv) {
    int i;
    for (i = 1; i < argc; i++) {
        if (!strcmp(str, argv[i])) {
            if (i == argc - 1) {
                printf("No argument given for %s\n", str);
                exit(1);
            }
            return i;
        }
    }
    return -1;
}

// 得到文件大小
long get_file_zize(char *filename) {
    struct stat64 statbuf;
    stat64(filename, &statbuf);
    long size = statbuf.st_size;
    return size;
}

// 分割
int train_test_split(char file[], float size, int header, int shuffle, int seed, int verbose_in) {
    FILE *fin, *fout_tr, *fout_te;

    char train_file[100] = "train_";
    char test_file[100] = "test_";
    strcat(train_file, file);
    strcat(test_file, file);

    long all_size = get_file_zize(file);
    long tr_size = (long) (get_file_zize(file) * (1 - size)) + 1024;
    long te_size = (long) (get_file_zize(file) * (size)) + 1024;

    register int is_train = 0;
    register int verbose = verbose_in;
    register int threshold = (int) (size * RAND_MAX);

    long tr_pos = 0;
    long te_pos = 0;
    long i = 0;
    int num_lines = 0;

    if (verbose > 0)
        printf("file size: %ld\n", all_size);


    char *all_buffer = (char *) malloc(all_size);
    char *tr_buffer = (char *) malloc(tr_size);
    char *te_buffer = (char *) malloc(te_size);

    fin = fopen(file, "r");
    if (fin == NULL) {
        fprintf(stderr, "Unable to open file %s.\n", file);
        return 1;
    }

    fout_tr = fopen(train_file, "w+");
    if (fout_tr == NULL) {
        fprintf(stderr, "Unable to open file %s.\n", train_file);
        return 1;
    }
    fout_te = fopen(test_file, "w+");
    if (fout_te == NULL) {
        fprintf(stderr, "Unable to open file %s.\n", test_file);
        return 1;
    }

    //  Read file
    if (verbose == 2)
        printf("reading file %s ...\n", file);
    fread(all_buffer, 1, all_size, fin);
    if (verbose == 2)
        printf("finished reading.\n");

    // Read lines
    srand(seed);
    if (rand() > threshold)
        is_train = 1;
    else
        is_train = 0;
    if (verbose > 0) {
        printf("begin to read line.\n");
    }
    int i_pos = i;
    if (header > 0) {
        while (all_buffer[i_pos] != '\n') {
            tr_buffer[tr_pos++] = all_buffer[i_pos];
            te_buffer[te_pos++] = all_buffer[i_pos];
            ++i_pos;
        }
        tr_buffer[tr_pos++] = '\n';
        te_buffer[te_pos++] = '\n';
    }
    for (i = i_pos; i < all_size; i++) {
        if (all_buffer[i] == '\n') {
            ++num_lines;
            if (verbose > 0) {
                if ((num_lines + 1) % 2000000 == 0)
                    printf("have read %d lines.\n", num_lines + 1);
            }
            if (is_train)
                tr_buffer[tr_pos++] = '\n';
            else
                te_buffer[te_pos++] = '\n';
            if (rand() > threshold)
                is_train = 1;
            else
                is_train = 0;
        } else {
            if (is_train)
                tr_buffer[tr_pos++] = all_buffer[i];
            else
                te_buffer[te_pos++] = all_buffer[i];
        }
    }

    // Set string end
    tr_buffer[tr_pos++] = '\n';
    tr_buffer[tr_pos] = '\0';
    te_buffer[te_pos++] = '\n';
    te_buffer[te_pos] = '\0';

    // Save to File
    if (verbose == 2)
        printf("writing file %s ...\n", train_file);
    fwrite(tr_buffer, 1, strlen(tr_buffer), fout_tr);
    if (verbose == 2)
        printf("write finished.\n");
    if (verbose == 2)
        printf("writing file %s ...\n", test_file);
    fwrite(te_buffer, 1, strlen(te_buffer), fout_te);
    if (verbose == 2)
        printf("write finished.\n");
}

int main(int argc, char **argv) {
    int i;
    char file[1024] = {0};
    float size = 0.2;
    int header = 0;
    int shuffle = 1;
    int seed = 1;
    int verbose = 2;

    if (argc == 1) {
        printf("Train Test Split\n");
        printf("Usage options:\n");
        printf("\t-file\n");
        printf("\t\tthe file to be split.\n");
        printf("\t-size <float>\n");
        printf("\t\ttest size.\n");
        printf("\t-header <int>\n");
        printf("\t\t0: no header, 1: with header.\n");
        printf("\t-shuffle <int>\n");
        printf("\t\t0: shuffle, 1: not(NOT IMPLEMENTED NOW!).\n");
        printf("\t-seed <int>\n");
        printf("\t\trandom seed (default 0).\n");
        printf("\t-verbose <int>\n");
        printf("\t\t0, 1, or 2 (default)\n");
        printf("\nExample usage:\n");
        printf("./train_test_split -file data.txt -size 0.2 -shuffle 1 -verbose 2\n\n");
    } else {
        if ((i = find_arg((char *) "-file", argc, argv)) > 0) strcpy(file, argv[i + 1]);
        if ((i = find_arg((char *) "-size", argc, argv)) > 0) size = atof(argv[i + 1]);
        if ((i = find_arg((char *) "-shuffle", argc, argv)) > 0) shuffle = atoi(argv[i + 1]);
        if ((i = find_arg((char *) "-header", argc, argv)) > 0) header = atoi(argv[i + 1]);
        if ((i = find_arg((char *) "-seed", argc, argv)) > 0) seed = atoi(argv[i + 1]);
        if ((i = find_arg((char *) "-verbose", argc, argv)) > 0) verbose = atoi(argv[i + 1]);

        if (verbose == 2) {
            printf("Recieved the args:\n");
            printf("file:\t\t%s\n", file);
            printf("size:\t\t%f\n", size);
            printf("header:\t\t%f\n", header);
            printf("shffle:\t\t%d\n", shuffle);
            printf("seed:\t\t%d\n", seed);
            printf("verbose:\t%d\n", verbose);
            printf("---------------------------------\n");
        }
        train_test_split(file, size, header, shuffle, seed, verbose);
    }
    return 0;
}
