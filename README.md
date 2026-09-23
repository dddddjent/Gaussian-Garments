# Gaussian garments: Reconstructing simulation-ready clothing with photorealistic appearance from multi-view video

### <img align=center src=./static/icons/project.png width='32'/> [Project](https://ribosome-rbx.github.io/Gaussian-Garments/) &ensp; <img align=center src=./static/icons/paper.png width='24'/> [Paper](https://arxiv.org/abs/2409.08189) &ensp;  

This is a repository for the paper [**"Gaussian garments: Reconstructing simulation-ready clothing with photorealistic appearance from multi-view video"**](https://ribosome-rbx.github.io/Gaussian-Garments/) (3DV2025).

This repository contains the implementation of the three stages described in the paper:
1. Geometry reconstruction (`s1_initiallisation.py`)
2. Garment registration (`s2_registration.py`)
3. Appearance reconstruction (`s3_appearance.py`)

This Readme file describes steps to 
* [Install the environment](#installation) 
* [Prepare input data for garment reconstruction](#data-preparation)
* [Create new Gaussian garments](#creating-new-gaussian-garments), and
* [Render the sequences of garment motions](#inference)

Unfortunately, we can not share the full data used in the paper. Hence, we have prepared the code for running Gaussian Garments algorithm on [ActorsHQ](https://actors-hq.com/) dataset. Notably, we found that the first stage used in the paper does not produce satysfying results on ActorsHQ, so we have prepared an alternative solution in the repository [ActorsHQ-for-Gaussian-Garments](https://github.com/hlimach/ActorsHQ-for-Gaussian-Garments).

**Follow instructions in [ActorsHQ-for-Gaussian-Garments](https://github.com/hlimach/ActorsHQ-for-Gaussian-Garments) to convert data from ActorsHQ into our format and reconstruct the garment mesh.** Then, you can continue from the stage 2 in this repository.

## Installation
The workspace uses a dedicated **gaugar** environment (Python 3.11,
PyTorch 2.11/CUDA 13, Blender 4.4.0, COLMAP 4.2) for the RTX 5080.
It is independent of `mpmavatar`. To create it from scratch, use `setup.sh`:
```bash
bash setup.sh
```
The script builds PyTorch3D, Simple-KNN, the depth/alpha Gaussian rasterizer,
and both StyleUNet CUDA extensions. COLMAP GPU option names are updated for 4.2.
Blender's 4.4.0 wheel has an incorrect `cp39` metadata tag, so `pip check` reports
that package despite successful Python 3.11 import. Its metadata is left intact.
Use separate processes for the native stages; importing PyMeshLab before Blender
in one interpreter conflicts over Embree. Native stage imports/loaders, CUDA
forward/backward, GPU capture and COLMAP GPU SIFT passed in `gaugar`;
[validation scope](../data/outputs/clothtransformer_gaussian_garments_validation/environment.json).
Native frame-0 initialization has now been attempted on both generated exports;
both stop with zero verified COLMAP matches. Training remains unrun. See the
[current experiment status and commands](../README.md#gaussian-garments-experiments).

After the installation, you can activate the environment with
```bash
conda activate gaugar
```

## Data Preparation
This repository assumes three key data locations:
* Input folder
* Output folder
* Auxilliary data folder

Refer to [DataPreparation.md](DataPreparation.md) for the instructions on how to set up these folders.

Then, your first step is to set paths to these three locations in `utils/defaults.py`:
```python
DEFAULTS['data_root'] = '/path/to/input/folder'
DEFAULTS['output_root'] = '/path/to/output/folder'
DEFAULTS['aux_root'] = '/path/to/auxilliary/data/folder'
```

For this workspace's generated exports, use `run.py --data ... --output ...
--stage initialize|template|register|appearance` instead. It reads the subject
from the manifest and passes per-process paths to each native script through
`GAUGAR_DATA_ROOT`, `GAUGAR_OUTPUT_ROOT`, and `GAUGAR_AUX_ROOT`.
Full [commands and current blockers](../README.md#gaussian-garments-experiments)
are in the workspace README. The original scripts still accept the defaults above.

For ClothTransformer frame 0, `--stage initialize --initialization gt-mesh`
optionally supplies the source garment mesh with existing UVs and image-sampled
colors. Use a separate output such as `output/ClothTransformer/sim_00000_gt_init`.
COLMAP remains the default; original initialization/registration/appearance
scripts are unchanged. The GT option applies only to initialization, supports
fitted and raw body exports, and uses no other cloth frames or held-out images.
The fitted-body `sim_00000_gt_init` assets passed source geometry/UV and native
GPU loader/render-gradient checks; full training remains unrun. Commands and
reports are in the [workspace experiment section](../README.md#gaussian-garments-experiments).

This checkout is configured for the prepared ClothTransformer `sim_00000`
training input and separate `output/ClothTransformer` results. SMPL-X auxiliary
assets are under `../data/GaussianGarments/auxiliary`. See the
[generation commands](../clothtransformer_avatar/README.md#gaussian-garments-preparation).

### Raw body input

`run.py` accepts `--body-model raw` ClothTransformer exports. Native stages discover
`body_mesh/` separately from calibrated camera folders. Registration retains all
raw vertices/faces; SMPL-X hand segmentation applies only to `smplx/` inputs.
Raw mode needs no auxiliary body models or hand labels. Appearance AO imports the
raw body with the same axes as the garment; missing bodies fail explicitly.
AO and normal maps are reused only when the recorded body/garment paths, file
timestamps/sizes, and bake settings match. Existing maps without records are
rebaked automatically. Per-frame locks serialize concurrent data-loader requests;
first use and cache hits read the same saved PNGs. Records live in
`stage2/<sequence>/texture/cache/`; no hashes are used.
The export's earlier `consumer_status` records its generation-time state.

Activate `gaugar` and run each stage separately from this directory:

```sh
DATA=../data/GaussianGarments/ClothTransformer/sim_00000_raw
OUT=output/ClothTransformer/sim_00000_raw
python run.py --data "$DATA" --output "$OUT" --stage initialize --template-frame 0
# Continue only after native initialization succeeds and stage1/template.obj
# has been manually UV-unwrapped to stage1/template_uv.obj.
python run.py --data "$DATA" --output "$OUT" --stage template --template-frame 0
python run.py --data "$DATA" --output "$OUT" --stage register --template-frame 0
python run.py --data "$DATA" --output "$OUT" --stage appearance --template-frame 0
```

2026-09-19: two input/launcher tests, all 100 raw body meshes, collision gradients
on the RTX 5080, and first/last train/held-out appearance loading and AO baking
passed. Baking used a temporary synthetic UV triangle (32×32, eight samples),
not a reconstructed garment. Both fitted export loaders also passed regression
checks. [Collision report](../data/outputs/gaussian_garments_raw_validation/collision.json),
[AO report](../data/outputs/gaussian_garments_raw_validation/appearance.json).
Raw frame-0 initialization reaches COLMAP but still has zero verified matches
across 28 pairs. No template or trained checkpoint was produced; temporary
outputs were removed. Full registration/appearance training and ContourCraft
physical prediction remain unrun.


## Creating new Gaussian garments
To reconstruct a new Gaussian garment from multi-view videos, you'll need to run three stages: (1) initialize the geomentry from a single frame, (2) register this geometry to all multi-view videos, and (3) train an appearance model to reconstruct the garment's appearance. Here we'll describe each of these steps.

Before running the reconstruction, make sure you have followed instructions in [DataPreparation.md](DataPreparation.md) and your input and auxilliary folders hava the same structure as described there.

All three steps share the same argument names:
* `-s/--subject`: name of the subject in the input folder (`DEFAULTS.data_root/*subject_id*`)
* `-so/--subject_out`: name of the directory in the output folder where the results will be stored (`DEFAULTS.output_root/*subject_out_id*`). If not set, the same name as in `--subject` is used
* `-q/--sequence`: name of the sequence to use. There may be more than one sequence for one subject. (`DEFAULTS.data_root/*subject_id*/*sequence_id*`)
* `-tf/--template_frame`: the number of the template frame in the sequence. By default `-tf 0` is used.

### Step 1: Geometry initialization
*Note: if you want to run Gaussian Garments for ActorsHQ or another dataset for which the 1st stage described in the paper produces unsatisfying results, refer to [ActorsHQ-for-Gaussian-Garments](https://github.com/hlimach/ActorsHQ-for-Gaussian-Garments) repository. After reconstructing the geometry with that repo, procede directly with Step 2.*

Assuming you have correct structure of the input folder, to reconstruct the garment geometry from a template frame, run:
```bash
python s1_initialisation.py -s *subject_id* -q *sequence_id* -tf *template_frame_id*
```
For the template frame, use the one where the whole surface of the garment is fully visible and where the pose is similar to the first frame of each sequence.

This script will generate the folder `DEFAULTS.output_root/*subject_id*/stage1`, where the reconstructed template will be stored in `template.obj`.

The final action in this step is to create a UV unwrapping for the template mesh. **This have to be done manually.** You can use Blender and follow [this YouTube guide](https://www.youtube.com/watch?v=LqJGD6yjlDE). **Save the .obj file of the mesh with the UV unwrapping to `DEFAULTS.output_root/*subject_id*/stage1/template_uv.obj`**

### Step 2: Garment registration
Once you have a garment template with uv unwrapping, you can register it to the multi-view videos. This process has two substeps.

First, we need to initialize the appearance of the garment. You can do this with this script:

*(Note that in the paper appearance initialization is described as a part of stage 1)*
```bash
python s2_registration.py -s *subject_id* -q *sequence_id*  -tf *template_frame_id*
```
You'll need to use the same sequence and template frame as in Step 1.

This will optimize the initial appearance and geometry for a single frame and save them into `DEFAULTS.output_root/*subject_id*/stage2/Template/`

Then, register this template to all sequences of this subject:
```bash
python s2_registration.py -s *subject_id* -q *sequence_id* 
python s2_registration.py -s *subject_id* -q *sequence_id2*
... 
```
This will produce registered meshes for each frame and save them into `DEFAULTS.output_root/*subject_id*/stage2/*sequence_id*/meshes/`. To check if it is running correctly, see the renders in `DEFAULTS.output_root/*subject_id*/stage2/*sequence_id*/renders/`.

Registering geometry to one frame takes a few minutes.


### Step 3: 
Finally, optimize the albedo texture and the appearance model using all registered sequences:
```bash
python s3_appearance.py -s *subject_id* 
```
This will store the training checkpoint into `DEFAULTS.output_root/*subject_id*/stage3/ckpt/`

### Step 4 and creating trajectories:
In step 4, we optimize the behavior of the garment by finetuning a ContourCraft graph neural network.

The workspace now provides `run.py --stage simulation-fit`, backed by
[`ContourCraft/fit_gaussian_garments.py`](../ContourCraft/fit_gaussian_garments.py).
It uses `train.py` with `configs/finetune/base.yaml` and the authors' garment
import/relaxation for fitted SMPL-X inputs; raw inputs use the Stage-1 garment
reference automatically. No notebook is needed. The original GNN, losses,
material/rest-edge optimization, and alternating CMU training batches are retained.
See the [author notebook](https://github.com/Dolorousrtur/ContourCraft/blob/main/GaussianGarments.ipynb)
for the original procedure.

For a fresh installation, create the separate `ccraft` environment with
`bash ContourCraft/setup.sh` from the workspace root, then activate `ccraft`. The setup targets RTX 5080 with
Python 3.10 and CUDA 13. The [author auxiliary data and pretrained checkpoint](https://github.com/Dolorousrtur/ContourCraft/blob/main/INSTALL.md#download-data)
belong in `data/ContourCraft/`. Licensed `SMPL_{MALE,FEMALE}.pkl` and
`SMPLX_{MALE,FEMALE,NEUTRAL}.npz` models belong under
`aux_data/body_models/{smpl,smplx}/`; provide the **AMASS CMU SMPL** motion directory
through `--cmu-root`. The original regularization batches require it.

From `Gaussian-Garments`, the launcher can stay in `gaugar`; select the separate
`ccraft` interpreter explicitly for simulation fitting:

```sh
DATA=../data/GaussianGarments/ClothTransformer/sim_00000
OUT=output/ClothTransformer/sim_00000
CMU=../../datasets/AMASS/CMU
python run.py --data "$DATA" --output "$OUT" --stage simulation-fit --template-frame 0 \
  --simulation-python /home/ljl/miniforge3/envs/ccraft/bin/python --ccraft-data ../data/ContourCraft \
  --cmu-root "$CMU"
```

Complete all training registrations first. Stage 3 appearance is not required
for fitting. `--prepare-only` exports training arrays/config without garment
relaxation or optimization and does not require `--cmu-root`; run the same command
without that flag to fit. `--steps` sets the total budget since initialization
(default: 1,000 combined batches; even and at least 2). `--checkpoint` selects
the pretrained starting model. To continue fitted state, use:

```sh
python run.py --data "$DATA" --output "$OUT" --stage simulation-fit \
  --simulation-python /home/ljl/miniforge3/envs/ccraft/bin/python --cmu-root "$CMU" \
  --resume --save-every 1
```

`--resume` selects the latest numeric checkpoint; `--resume PATH` selects one
explicitly. It restores network/material/rest-edge parameters, both optimizers,
schedulers, completed step and saved random states, retaining the saved target
unless `--steps` supplies a new total. Data iterators restart with fresh sampling;
this does not replay the original data order exactly. New checkpoints are written
atomically every 10 completed batches by default; `--save-every 1` saves each batch.
Ctrl+C finishes the current sequence batch and saves before exiting. Existing
processes started before this change retain their original stop/save behavior.
Each launch creates a new offline W&B run. A checkpoint is required to resume;
unsaved work cannot be recovered. Older checkpoints are rejected if newer steps
already exist in this output.

Outputs live in `$OUT/stage4/`: `smplx/train.npz` (or `body_mesh/train.npz` for raw bodies), `registrations/train.pkl`,
`garment_dicts/`, `finetune.yaml`, and `checkpoints/step_*.pth`. Metrics are logged
locally in W&B offline mode. Only Stage-2 training registrations supervise fitting;
held-out observations are excluded and the validation CSV is empty. Target
timesteps use the export FPS (25 here), while original CMU batches keep the
authors' timestep. Full axis-angle hands, hand means, wrist poses, and chronological
initial frames are preserved for fitted SMPL-X inputs. Raw `body_mesh/` exports
use their original vertex correspondence and topology, without SMPL fitting,
skinning, anatomical hand masks, or parametric garment relaxation. The garment
reference is automatically `stage1/template_uv.obj`, following the authors'
mesh-input convention:

```sh
DATA=../data/GaussianGarments/ClothTransformer/sim_00000_raw
OUT=output/ClothTransformer/sim_00000_raw_gt_init
python run.py --data "$DATA" --output "$OUT" --stage simulation-fit \
  --simulation-python /home/ljl/miniforge3/envs/ccraft/bin/python --cmu-root "$CMU"
```

The Stage-1 reference is recorded in `preparation.json`. Original AMASS
regularization and losses remain unchanged.
Raw evaluation uses the same `--stage evaluate` command with the raw `DATA`/`OUT`;
it joins body motion across the split and initializes cloth from training frames
0/1 only. All 24 CPU conversion, native loader, evaluation-sample, and launcher
checks passed; [all 100 raw body frames](../data/outputs/contourcraft_raw_validation/body.json)
preserve source coordinates exactly. A [bounded GPU check](../data/outputs/contourcraft_raw_validation/runtime/report.json)
passed one target-fitting optimizer step, checkpoint reload, and four prediction
steps on real registrations. Full alternating AMASS fitting and held-out accuracy
evaluation remain unrun.

2026-09-20: `ccraft` is installed; author pretrained/auxiliary assets and licensed
body-model links are ready in `data/ContourCraft`. Ten launcher/conversion tests
passed. [GPU runtime checks](../data/outputs/contourcraft_validation/runtime.json)
passed, including actual triangle collisions and strict author checkpoint loading;
[conversion checks](../data/outputs/contourcraft_validation/conversion.json) preserve
the existing fitted body geometry. AMASS CMU is now installed at
`../../datasets/AMASS/CMU`; all 46 required motion files passed schema checks.
The production `stage4/` inputs are prepared from all 80 registered training
frames. Full fitting remains unrun.


### Inference
For prepared ClothTransformer/D-Garment experiments, use the calibrated evaluation
adapter in `gaugar` after ContourCraft fitting:

```sh
python run.py --data ../data/GaussianGarments/ClothTransformer/sim_00000 \
  --output output/ClothTransformer/sim_00000 --stage evaluate \
  --simulation-python /home/ljl/miniforge3/envs/ccraft/bin/python
```

Add `--evaluate-from-start` to render frame 0 through the end instead of only the
held-out suffix. Both ranges use continuous simulation initialized with the first
two training registrations, frozen fitted parameters, and future body motion only.
The default checkpoints are the highest saved ContourCraft step and completed
appearance epoch; `--checkpoint PATH` and `--appearance-checkpoint PATH` override them.
Both fitted SMPL-X and raw-body exports are supported. For raw evaluation, use
the raw `DATA`/`OUT` paths above; texture baking uses the same raw collider.

Each fresh `evaluation/{held_out,full_sequence}/` destination contains predictions
and separate `gt_lighting/<camera>/` and `fitted_appearance/<camera>/` trees with
prediction/reference PNGs and `pred.mp4`, `gt.mp4`, `comparison.mp4` at source FPS.
Comparison layout is prediction left, capture GT right. Capture materials/lights
and fitted Gaussian appearance are rendered on the same predicted garment;
the visible body is the original source surface. Two-frame/eight-camera renderer
checks passed; full fitted simulation evaluation awaits a ContourCraft checkpoint.

For the authors' standalone trajectory preview:
To render a dynamic sequence of Gaussian garment geometries, use `inference.py` script:
```bash
python inference.py --traj_path *trajectory_file*.pkl --output_path *directory_to_store_renders*
``` 

Here, `trajectory_file` is a path to a `.pkl` file containing trajectories of the garments and the body meshes. They follow the structure of the trajectory files produced by [ContourCraft](https://github.com/dolorousrtur/ContourCraft) and contain a dictionary with the following elements:

* `pred`: 3D positions of garment vertices in each frame, np.array of shape [N, V, 3], where N is the number of frames and V is the number of the garment vertices.
* `cloth_faces`: faces of the garment meshes, np.array of shape [F, 3]
* `obstacle`: 3D positions of body vertices in each frame, np.array of shape [N, B, 3], where N is the number of frames and B is the number of the body vertices.
* `obstacle_faces`: faces of the body mesh, np.array of shape [Fb, 3]
* `garment_names`: a list of subject names for the garments used in the trajectory. The subject names should correspond to the folders in your `DEFAULTS.output_root` directory. 

A trajectory file may contain a trajectory for a multi-garment outfit comprised of several garments. In this case, 

* `pred` is a concatenation of vertex positions for all garments, 
* `cloth_faces` is the concatenation of their faces with properly renumerated vertex ids, and 
* `garment_names` is the list of subject names in the same order as the garments are concatenated in.

The script `inference.py` will check the  `garment_names` list and load the corresponding checkpoints stored in your `DEFAULTS.output_root` directory. 

See the link to examples of the trajectory files below.

#### Render sequences from the supplementary video
You can download the trajectory files used in the supplementary video [here](https://drive.google.com/file/d/1VoCmCfL-YmWL4BsgcdN8gxcEt3djJp-v/view?usp=drive_link).

To render them, you will need to download checkpoints of the Gaussian Garments used in the paper [here](https://drive.google.com/file/d/1EnmIQQ3BIN9nqykhEAOv18R5mH1eq5fx/view?usp=sharing). Unpack them to your `DEFAULTS.output_root` so that its structure is:

```
DEFAULTS.output_root/
├── 00122_outer/
├── 00134_upper/
└── ...
```

Then, you will be able to render the trajectories with the command above.
