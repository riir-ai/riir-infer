# Analytic key census of the three f16-B twin caches over tails 1..=chunk.
# Key formulas (riir-infer @HEAD): shadow pool = p*w*2 bytes (w in rotate widths);
# twin output_cache = p*m*4 bytes (m = every twin-routed GEMM's out dim);
# input_cache = p*n*2 (cast fallback only; 0 new keys when every served site has a shadow).
W_SHADOW=[5120,6144,17408]               # n (AttnQ/GdnQkv/FfnGate), q_dim=v_dim (AttnOut/GdnOut), mlp (FfnDown)
M_OUT=[12288,1024,5120,10240,6144,48,17408]  # AttnQ, AttnKv, *Out/Down, GdnQkv, GdnZ, GdnA/B, FfnGate/Up
def census(chunk, bucket=None):
    b=(lambda p:p) if bucket is None else bucket
    # exact = the pre-1238 key (byte length); bucketed = the landed key
    # (row_bytes, rows.next_power_of_two()), buffer = row_bytes * bucket.
    if bucket is None:
        sh={2*p*w for p in range(1,chunk+1) for w in W_SHADOW}
        out={4*p*m for p in range(1,chunk+1) for m in M_OUT}
        return len(sh),sum(sh),len(out),sum(out)
    sh={(2*w,b(p)) for p in range(1,chunk+1) for w in W_SHADOW}
    out={(4*m,b(p)) for p in range(1,chunk+1) for m in M_OUT}
    return len(sh),sum(r*k for r,k in sh),len(out),sum(r*k for r,k in out)
pow2=lambda p:1<<(p-1).bit_length()
for chunk in (128,512,4096):
    for name,bk in (("exact",None),("pow2",pow2)):
        ks,bs,ko,bo=census(chunk,bk)
        full=sum(2*chunk*w for w in W_SHADOW)+sum(4*chunk*m for m in M_OUT)
        print(f"chunk {chunk:5} {name:5}: shadow {ks:6} keys {bs/2**30:8.2f} GiB | output {ko:6} keys {bo/2**30:8.2f} GiB | total {(bs+bo)/2**30:8.2f} GiB  (fixed-width floor {full/2**30:.3f} GiB)")
