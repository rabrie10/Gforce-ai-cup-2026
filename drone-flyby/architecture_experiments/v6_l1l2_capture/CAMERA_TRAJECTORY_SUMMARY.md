# Second Hosted V6 Run: Metadata-Only Trajectory

Source: `/workspace/logs/endpoint_v6_public_9053.log.run2_424a4205_1789860870`

This is request metadata, not image data. It must not be used to infer hidden
objects or ground-truth locations.

- Requests: 185
- Received views: L0=1, L1=101, L2=83
- Received L1 center bounds: x=960..2880, y=540..1620
- Received L2 center bounds: x=480..3240, y=270..1890
- Received level transitions: L0->L1=1, L1->L1=59, L1->L2=42, L2->L1=41, L2->L2=41
- Requested view levels: L1=102, L2=83
- Logged endpoint latency: minimum 27.53 ms, mean 52.19 ms, maximum 174.84 ms

The log records `view_level`, `view_center`, and `requested_view`. It does not
contain image payloads, encoded images, source pixels, or per-request image
files. The existing `/workspace/hosted-v3` images are a separate sequence and
are all L0.