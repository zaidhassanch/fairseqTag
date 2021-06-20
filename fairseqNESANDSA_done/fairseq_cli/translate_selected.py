from fairseq import options, distributed_utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf

from fairseq_cli.interactive_selected import main

def translate_sents(cfg, modelZaid):
    cfg.interactive.input = "testSentences.txt"
    cfg.interactive.buffer_size = 1

    distributed_utils.call_main(cfg, main, model_zaid=modelZaid)


# if __name__ == "__main__":
#     model = None
#     translate_sents(model)
