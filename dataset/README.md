# Dataset

`g1/CMU/` contains a selected subset of the CMU motions from
[AMASS Retargeted for G1](https://huggingface.co/datasets/ember-lab-berkeley/AMASS_Retargeted_for_G1)
(AMASS motions retargeted to the Unitree G1). During teacher training these
clips replay the upper-body arm motions while the policy controls the legs
and waist.

## Format

Files are `<subject_id>/<clip>_jpos.npy` with shape `(T, 36)`:
columns `0:3` root world position (m), `3:7` root quaternion (x, y, z, w),
`7:36` the 29 joint positions (rad) in Unitree G1 XML order. This repository
reads only the 14 arm columns (`22:36`).

Loaded by `legged_lab/utils/amass.py` and replayed on the 14 arm joints by all
simulation entry points (training and play). Deployment does not use it; the
arms are held at configured default positions on hardware.

## License and provenance

The motion files under `g1/CMU/` were manually selected from the CMU subset of
[AMASS Retargeted for G1](https://huggingface.co/datasets/ember-lab-berkeley/AMASS_Retargeted_for_G1), and are included only for users’ convenience.
They are not covered by this repository's BSD-3-Clause software license.
Use and redistribution remain subject to the applicable original dataset terms.
No additional rights to these files are granted by this repository.
