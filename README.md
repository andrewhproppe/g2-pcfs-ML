
## Changelog

- Transforms pipeline now work on dictionary batches for easier referencing
- Cleaned up dataset and datamodule implementation  in `g2_pcfs/pipeline/image_data.py`
- New model types within the `g2_pcfs.models.ode` module:
	- `g2_pcfs.models.ode.ode` contains the high level PyTorch Lightning
	- `g2_pcfs.models.ode.ode_models` contains implementations of models that slot in
- Working training script in `scripts/train_neuralode.py`
	- Nominally working, but adjoint training is very unstable and can stall because number of evaluations blow up

## TODO

- Test "real" architectures; MLP seems to work perfectly well but maybe we can improve on it
- Develop predict pipeline for inference
