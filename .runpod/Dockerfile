# RunPod Hub build image for the Vivijure Wan cast-training worker.
#
# WHY THIS IS NOT deploy/Dockerfile: the production build COPYs deploy/train-bins/, ~68 GB of
# Wan 2.2 base weights that are staged at CI time (scripts/download_wan_weights.py +
# deploy/bake_layers.py in .github/workflows/build-image.yml) and are gitignored. Nothing in the
# repo tree can reproduce them, so the Hub builder cannot build deploy/Dockerfile at all.
# Hub therefore consumes the SAME published artifact the production endpoint runs: the public
# GHCR image, pinned by tag AND digest. CMD, ENV, and both conda envs are inherited.
#
# REPIN ON EVERY IMAGE RELEASE: bump the tag AND the @sha256 below after build-image.yml
# publishes a new train-<version>, then cut a GitHub release so Hub re-indexes.
# scripts/check_hub_base_pin.sh (run by .github/workflows/hub-pin-check.yml) fails the build if
# the pin does not resolve on GHCR or if tag and digest disagree.
FROM ghcr.io/skyphusion-labs/vivijure-wan-train:train-0.1.3@sha256:ce10aa2d531f2d870aa8db868c62fd0cfea3caff68346c0c4e9b636261fd2f22
