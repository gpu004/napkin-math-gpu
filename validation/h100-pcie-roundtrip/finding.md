Outside tolerance, about 5.9× slower than predicted. This is a miss in the
model's assumptions, not one to tune away. The profile's 54–55 GB/s host link
is a property of the host it was measured on, not of the H100.

Confirmed from the committed evidence:

- NVBandwidth measured 14.48 GB/s upload and 6.77 GB/s download, about 27% and
  12% of the profile's bandwidth, both reported as `Passed`.
- nvidia-smi reported the link at Gen5 x16 (current and max) throughout the
  copies, so the match criterion on the link held. The link generation did not
  predict the bandwidth.
- The host ran driver 610.57.04 and nvidia-smi could not read its PCI bus id
  (`[Unknown Error]` in run.json). The profile host ran 580.95.05.
- [followup.json](validation/h100-pcie-roundtrip/followup.json): two further Modal hosts, one `H100!` (driver
  610.57.04) and one `H100!:2` (580.95.05). Both reported a Gen4 x16 link and
  ran 27.6 GB/s up and 28.5 GB/s down at both 512 MiB and 1 GiB buffers. The
  buffer size is therefore not the cause, and the same `H100!` request has
  landed on hosts with at least three different host-link bandwidths: 55, 28
  and 14/6.8 GB/s.

Hypothesis, not verified: the observation host's copy path is shared or
degraded, for example a PCIe switch contended by another tenant's GPU, or
IOMMU/virtualization overhead. Download running at half of upload is unlike
every other host measured.

Consequence for napkin: a host-copy estimate is only as good as a bandwidth
measurement from the same host. On Modal, check the host with NVBandwidth in
the same container before trusting an `h2d`/`d2h`-bound estimate.
