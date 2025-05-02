from src.dr_alns.Trainer import Trainer, get_parameters

from psp_AlnsEnv import SMJSPAlnsEnv

if __name__ == "__main__":
    # Training the model
    config = get_parameters("pspAlnsEnv.yml")
    env = SMJSPAlnsEnv(config)
    trainer = Trainer(env=SMJSPAlnsEnv, config=config)
    trainer.create_model()
    trainer.train()
