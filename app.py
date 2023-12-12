from audio_file_feature_extractor import AudioFileFeatureExtractor
from gradio_ui_generator import GradioUIGenerator
from model_handler import ModelHandler
from urban_sound_dataset_handler import UrbanSoundDatasetHandler


def create_demo():
    dataset_handler = UrbanSoundDatasetHandler(regenerate=False)
    model_handler = ModelHandler(dataset_handler)
    extractor = AudioFileFeatureExtractor(model_handler, dataset_handler)
    gradio_ui = GradioUIGenerator(extractor, model_handler, dataset_handler)
    demo = gradio_ui.generate_demo()
    return demo


demo = create_demo()
demo.launch()
