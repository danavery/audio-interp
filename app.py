import logging

from audio_file_feature_extractor import AudioFileFeatureExtractor
from gradio_ui_generator import GradioUIGenerator
from model_handler import ModelHandler
from urban_sound_dataset_handler import UrbanSoundDatasetHandler

logging.basicConfig(
    format="%(asctime)s - %(filename)s: %(lineno)d - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.setLevel("INFO")


class App:
    def __init__(self):
        logger.info("starting usdh")
        self.dataset_handler = UrbanSoundDatasetHandler(regenerate=False)
        logger.info("starting model handler")
        self.model_handler = ModelHandler(self.dataset_handler)
        logger.info("starting feature extractor")
        self.audio_file_feature_extractor = AudioFileFeatureExtractor(
            self.model_handler, self.dataset_handler
        )
        logger.info("starting Gradio generator")
        self.gradio_ui = GradioUIGenerator(
            self.audio_file_feature_extractor,
            self.model_handler,
            self.dataset_handler,
            self.audio_file_feature_extractor,
        )


# app = App()
# ui_generator = app.gradio_ui
demo = App().gradio_ui.generate_demo()
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
