/* T1 power-sampler survey probe (issue 1220) — sudo-free IOReport access check.
 * The IOReport functions are NOT in the public IOKit tbd, so this probe dlopen()s
 * IOKit and dlsym()s the four functions it needs. Exit codes:
 *   0 = energy channels sampled sudo-free (deltas shown)
 *   2 = copy-channels nil          3 = subscription failed
 *   4 = samples nil                5 = no channels iterated
 *  10 = dlsym could not resolve the API (the IOReport-class candidate is DEAD
 *       sudo-free; the powermetrics sudoers entry is the remaining joule meter)
 */
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOReturn.h>
#include <stdbool.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

typedef CFMutableDictionaryRef (*copy_channels_fn)(CFStringRef, CFArrayRef, uint64_t, uint64_t, uint64_t);
typedef kern_return_t (*create_sub_fn)(void *, CFMutableDictionaryRef, CFMutableDictionaryRef *, uint64_t, CFTypeRef);
typedef CFMutableDictionaryRef (*create_samples_fn)(CFMutableDictionaryRef, CFMutableDictionaryRef, void *);
typedef kern_return_t (*merge_samples_fn)(CFMutableDictionaryRef, CFMutableDictionaryRef, void *);

/* channel iteration: the ABI takes a C BLOCK (^bool(IOReportChannelRef, void *)),
 * not a plain function pointer — the block is passed directly. */
typedef void *IOReportChannelRef;
typedef kern_return_t (*iterate_fn)(CFMutableDictionaryRef, bool (^block)(IOReportChannelRef, void *), void *);
typedef CFDictionaryRef (*chan_info_fn)(IOReportChannelRef);
typedef CFDictionaryRef (*chan_sample_fn)(IOReportChannelRef);

static void *kit = NULL;
static copy_channels_fn p_copy;
static create_sub_fn p_sub;
static create_samples_fn p_samples;
static merge_samples_fn p_merge;
static iterate_fn p_iterate;
static chan_info_fn p_info;
static chan_sample_fn p_sample;

static void *sym(const char *n) {
    void *p = dlsym(kit, n);
    if (!p) fprintf(stderr, "dlsym miss: %s\n", n);
    return p;
}

int main(int argc, char **argv) {
    const char *group_c = argc > 1 ? argv[1] : "Energy Model";

    kit = dlopen("/System/Library/Frameworks/IOKit.framework/IOKit", RTLD_LAZY);
    if (!kit) { printf("DLOPEN: %s\n", dlerror()); return 10; }
    p_copy = (copy_channels_fn)sym("IOReportCopyChannelsInGroup");
    p_sub = (create_sub_fn)sym("IOReportCreateSubscription");
    p_samples = (create_samples_fn)sym("IOReportCreateSamples");
    p_merge = (merge_samples_fn)sym("IOReportMergeSamples");
    p_iterate = (iterate_fn)sym("IOReportIterateChannelsWithBlock");
    p_info = (chan_info_fn)sym("IOReportChannelGetInfo");
    p_sample = (chan_sample_fn)sym("IOReportChannelGetSample");
    if (!p_copy || !p_sub || !p_samples || !p_merge || !p_iterate || !p_info || !p_sample) {
        printf("DLSYM: incomplete — the IOReport-class candidate is NOT reachable\n");
        return 10;
    }

    CFStringRef group = CFStringCreateWithCString(NULL, group_c, kCFStringEncodingUTF8);
    CFMutableDictionaryRef chans = p_copy(group, NULL, 0, 0, 0);
    if (!chans) { printf("COPY-CHANNELS: nil for group '%s'\n", group_c); return 2; }
    printf("COPY-CHANNELS: ok (group '%s')\n", group_c);

    CFMutableDictionaryRef subscribed = NULL;
    IOReturn st = p_sub(NULL, chans, &subscribed, 0, NULL);
    printf("SUBSCRIBE: status=0x%x (%s)\n", st, st == 0 ? "ok" : "FAILED");
    if (st != 0 || !subscribed) return 3;

    CFMutableDictionaryRef s0 = p_samples(subscribed, subscribed, NULL);
    usleep(200000);
    CFMutableDictionaryRef s1 = p_samples(subscribed, subscribed, NULL);
    printf("SAMPLES: s0=%s s1=%s\n", s0 ? "ok" : "nil", s1 ? "ok" : "nil");
    if (!s0 || !s1) return 4;
    p_merge(s1, s0, NULL); /* s1 = deltas between the two samples */

    /* The iteration callback is a C BLOCK (^bool(channel, ctx)) on this ABI. */
    __block int printed = 0;
    __block double total = 0;
    IOReturn it = p_iterate(s1, ^bool(IOReportChannelRef channel, void *ctx) {
        (void)ctx;
        CFDictionaryRef info = p_info(channel);
        CFDictionaryRef sample = p_sample(channel);
        char name[128] = "?", subgrp[128] = "?", unit[64] = "?";
        int64_t count = -1;
        if (info) {
            CFStringRef cn = CFDictionaryGetValue(info, CFSTR("channel-name"));
            CFStringRef cs = CFDictionaryGetValue(info, CFSTR("subgroup"));
            CFStringRef cu = CFDictionaryGetValue(info, CFSTR("unit-label"));
            if (cn) CFStringGetCString(cn, name, sizeof(name), kCFStringEncodingUTF8);
            if (cs) CFStringGetCString(cs, subgrp, sizeof(subgrp), kCFStringEncodingUTF8);
            if (cu) CFStringGetCString(cu, unit, sizeof(unit), kCFStringEncodingUTF8);
        }
        if (sample) {
            CFNumberRef c = CFDictionaryGetValue(sample, CFSTR("Channel Count"));
            if (c) CFNumberGetValue(c, kCFNumberSInt64Type, &count);
        }
        if (printed < 14) {
            printf("CH: %-52s [%s] unit=%-14s delta=%lld\n", name, subgrp, unit, (long long)count);
            printed++;
        }
        if (count > 0) total += (double)count;
        return true;
    }, NULL);
    printf("ITERATE: status=0x%x shown=%d totalDelta(shown subset)=%.0f\n", it, printed, total);
    return printed > 0 ? 0 : 5;
}
