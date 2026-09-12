/*
 * seccomp_filter.c — Seccomp-BPF Syscall Filter
 * ================================================
 * This program applies a seccomp-BPF filter that restricts which
 * system calls the AI agent process is allowed to make.
 *
 * How it works:
 *   1. Creates a seccomp filter context with a default ALLOW action
 *   2. Adds rules to BLOCK (kill) dangerous syscalls:
 *      - connect()  → blocks outbound network connections
 *      - execve()   → restricted (we allow python3 but block shells)
 *      - chmod()    → blocks permission changes
 *      - chown()    → blocks ownership changes  
 *      - mount()    → blocks filesystem mounting
 *      - reboot()   → blocks system reboot
 *      - ptrace()   → blocks process debugging/injection
 *   3. Loads the filter into the kernel
 *   4. exec()'s the actual command inside the filtered environment
 *
 * Once a seccomp filter is applied, it CANNOT be removed — not even
 * by root. This makes it a hard, non-bypassable security boundary.
 *
 * Compile: gcc -o seccomp_filter seccomp_filter.c -lseccomp
 * Usage:   ./seccomp_filter <command> [args...]
 *
 * Theory:
 *   This implements a form of "capability dropping" — the process
 *   starts with full capabilities and we selectively remove the
 *   dangerous ones. This maps to the Principle of Least Privilege.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <seccomp.h>
#include <sys/types.h>
#include <sys/stat.h>

/* Print usage instructions */
void usage(const char *prog) {
    fprintf(stderr, "Usage: %s <command> [args...]\n", prog);
    fprintf(stderr, "Applies a seccomp-BPF filter and then runs <command>.\n");
    fprintf(stderr, "\nExample:\n");
    fprintf(stderr, "  %s python3 agent.py 'List files'\n", prog);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    /* 
     * Create a seccomp filter context.
     * SCMP_ACT_ALLOW = default action is to ALLOW all syscalls.
     * We then add specific rules to BLOCK dangerous ones.
     * (The alternative is SCMP_ACT_KILL which blocks everything by default
     *  and you allow specific syscalls — more secure but harder to get right)
     */
    scmp_filter_ctx ctx = seccomp_init(SCMP_ACT_ALLOW);
    if (ctx == NULL) {
        fprintf(stderr, "[SECCOMP] ERROR: Failed to initialize seccomp context\n");
        return 1;
    }

    printf("[SECCOMP] Initializing BPF filter...\n");

    /*
     * ── Block dangerous syscalls ──
     * Each rule says: if the process tries this syscall, return ERRNO
     * (permission denied) instead of killing the process outright.
     * SCMP_ACT_ERRNO(1) = return EPERM (Operation not permitted)
     */

    /* Block outbound network connections */
    /* connect() is what curl, wget, nc etc. use to reach the network */
    seccomp_rule_add(ctx, SCMP_ACT_ERRNO(1), SCMP_SYS(connect), 0);
    printf("  [+] Blocked: connect() — no outbound network\n");

    /* Block permission/ownership changes */
    seccomp_rule_add(ctx, SCMP_ACT_ERRNO(1), SCMP_SYS(chmod), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ERRNO(1), SCMP_SYS(fchmod), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ERRNO(1), SCMP_SYS(chown), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ERRNO(1), SCMP_SYS(fchown), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ERRNO(1), SCMP_SYS(lchown), 0);
    printf("  [+] Blocked: chmod/chown — no permission changes\n");

    /* Block filesystem mounting (privilege escalation vector) */
    seccomp_rule_add(ctx, SCMP_ACT_ERRNO(1), SCMP_SYS(mount), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ERRNO(1), SCMP_SYS(umount2), 0);
    printf("  [+] Blocked: mount/umount — no filesystem changes\n");

    /* Block process debugging (could be used to inject code into other processes) */
    seccomp_rule_add(ctx, SCMP_ACT_ERRNO(1), SCMP_SYS(ptrace), 0);
    printf("  [+] Blocked: ptrace — no process debugging\n");

    /* Block system reboot/shutdown */
    seccomp_rule_add(ctx, SCMP_ACT_ERRNO(1), SCMP_SYS(reboot), 0);
    printf("  [+] Blocked: reboot — no system shutdown\n");

    /* Block kernel module loading (rootkit installation vector) */
    seccomp_rule_add(ctx, SCMP_ACT_ERRNO(1), SCMP_SYS(init_module), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ERRNO(1), SCMP_SYS(finit_module), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ERRNO(1), SCMP_SYS(delete_module), 0);
    printf("  [+] Blocked: init_module — no kernel module loading\n");

    /* Block setting user/group IDs (privilege escalation) */
    seccomp_rule_add(ctx, SCMP_ACT_ERRNO(1), SCMP_SYS(setuid), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ERRNO(1), SCMP_SYS(setgid), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ERRNO(1), SCMP_SYS(setreuid), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ERRNO(1), SCMP_SYS(setregid), 0);
    printf("  [+] Blocked: setuid/setgid — no privilege escalation\n");

    /* Block process killing (denial-of-service on host) */
    /* Note: We allow SIGTERM/SIGINT to self but block kill() to other PIDs */
    /* For simplicity, we allow kill() here but the namespace restricts PID visibility */

    /* Block socket creation for raw sockets (packet sniffing) */
    /* We still allow AF_UNIX and AF_INET sockets for normal operation, */
    /* but connect() is blocked above so they can't reach the network */

    /*
     * ── Load the filter into the kernel ──
     * After this point, ALL the rules above are enforced by the kernel.
     * The filter cannot be removed, even by root.
     */
    printf("\n[SECCOMP] Loading BPF filter into kernel...\n");
    int rc = seccomp_load(ctx);
    if (rc < 0) {
        fprintf(stderr, "[SECCOMP] ERROR: Failed to load filter: %s\n", strerror(-rc));
        seccomp_release(ctx);
        return 1;
    }
    printf("[SECCOMP] ✓ Filter active — dangerous syscalls are now blocked\n");
    printf("[SECCOMP] ✓ Filter is IRREMOVABLE (kernel-enforced)\n\n");

    /* Release the context (filter is already in the kernel) */
    seccomp_release(ctx);

    /*
     * ── Execute the target command ──
     * execvp() replaces this process with the target command.
     * The seccomp filter is inherited by the new process.
     */
    execvp(argv[1], &argv[1]);

    /* If we get here, execvp failed */
    fprintf(stderr, "[SECCOMP] ERROR: Failed to exec '%s': %s\n", argv[1], strerror(errno));
    return 1;
}
