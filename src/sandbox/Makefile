# Makefile for the Seccomp-BPF Filter
# Compile: make
# Clean:   make clean

CC = gcc
CFLAGS = -Wall -Wextra -O2
LDFLAGS = -lseccomp
TARGET = seccomp_filter

.PHONY: all clean

all: $(TARGET)

$(TARGET): seccomp_filter.c
	$(CC) $(CFLAGS) -o $(TARGET) seccomp_filter.c $(LDFLAGS)
	@echo "[BUILD] ✓ seccomp_filter compiled successfully"
	@echo "[BUILD] Usage: ./seccomp_filter <command> [args...]"

clean:
	rm -f $(TARGET)
	@echo "[BUILD] Cleaned"
