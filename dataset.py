import torch
from datasets import load_dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pytorch_lightning as pl

dataset = load_dataset("qanastek/MASSIVE", "en-US", data_dir='MASSIVE')

_INTENTS = ['audio_volume_other', 'play_music', 'iot_hue_lighton', 'general_greet', 'calendar_set', 'audio_volume_down', 'social_query', 'audio_volume_mute', 'iot_wemo_on', 'iot_hue_lightup', 'audio_volume_up', 'iot_coffee', 'takeaway_query', 'qa_maths', 'play_game', 'cooking_query', 'iot_hue_lightdim', 'iot_wemo_off', 'music_settings', 'weather_query', 'news_query', 'alarm_remove', 'social_post', 'recommendation_events', 'transport_taxi', 'takeaway_order', 'music_query', 'calendar_query', 'lists_query', 'qa_currency', 'recommendation_movies',
            'general_joke', 'recommendation_locations', 'email_querycontact', 'lists_remove', 'play_audiobook', 'email_addcontact', 'lists_createoradd', 'play_radio', 'qa_stock', 'alarm_query', 'email_sendemail', 'general_quirky', 'music_likeness', 'cooking_recipe', 'email_query', 'datetime_query', 'transport_traffic', 'play_podcasts', 'iot_hue_lightchange', 'calendar_remove', 'transport_query', 'transport_ticket', 'qa_factoid', 'iot_cleaning', 'alarm_set', 'datetime_convert', 'iot_hue_lightoff', 'qa_definition', 'music_dislikeness']
_TAGS = ['O', 'B-food_type', 'B-movie_type', 'B-person', 'B-change_amount', 'I-relation', 'I-game_name', 'B-date', 'B-movie_name', 'I-person', 'I-place_name', 'I-podcast_descriptor', 'I-audiobook_name', 'B-email_folder', 'B-coffee_type', 'B-app_name', 'I-time', 'I-coffee_type', 'B-transport_agency', 'B-podcast_descriptor', 'I-playlist_name', 'B-media_type', 'B-song_name', 'I-music_descriptor', 'I-song_name', 'B-event_name', 'I-timeofday', 'B-alarm_type', 'B-cooking_type', 'I-business_name', 'I-color_type', 'B-podcast_name', 'I-personal_info', 'B-weather_descriptor', 'I-list_name', 'B-transport_descriptor', 'I-game_type', 'I-date', 'B-place_name', 'B-color_type', 'B-game_name', 'I-artist_name', 'I-drink_type', 'B-business_name', 'B-timeofday', 'B-sport_type', 'I-player_setting', 'I-transport_agency', 'B-game_type', 'B-player_setting', 'I-music_album', 'I-event_name', 'I-general_frequency', 'I-podcast_name', 'I-cooking_type', 'I-radio_name', 'I-joke_type',
         'I-meal_type', 'I-transport_type', 'B-joke_type', 'B-time', 'B-order_type', 'B-business_type', 'B-general_frequency', 'I-food_type', 'I-time_zone', 'B-currency_name', 'B-time_zone', 'B-ingredient', 'B-house_place', 'B-audiobook_name', 'I-ingredient', 'I-media_type', 'I-news_topic', 'B-music_genre', 'I-definition_word', 'B-list_name', 'B-playlist_name', 'B-email_address', 'I-currency_name', 'I-movie_name', 'I-device_type', 'I-weather_descriptor', 'B-audiobook_author', 'I-audiobook_author', 'I-app_name', 'I-order_type', 'I-transport_name', 'B-radio_name', 'I-business_type', 'B-definition_word', 'B-artist_name', 'I-movie_type', 'B-transport_name', 'I-email_folder', 'B-music_album', 'I-house_place', 'I-music_genre', 'B-drink_type', 'I-alarm_type', 'B-music_descriptor', 'B-news_topic', 'B-meal_type', 'I-transport_descriptor', 'I-email_address', 'I-change_amount', 'B-device_type', 'B-transport_type', 'B-relation', 'I-sport_type', 'B-personal_info']


def index_to_intent(index):
    return _INTENTS[index]


def index_to_tag(index):
    return _TAGS[index]


def get_intent_slots():
    intent_slots = {}
    for data in dataset['train']:
        intent = data['intent']
        if intent not in intent_slots:
            intent_slots[intent] = set(data['ner_tags'])
        else:
            intent_slots[intent] = intent_slots[intent].union(
                set(data['ner_tags']))

    named_data = {}
    for intent, slots in intent_slots.items():
        named_data[intent] = 'slots: ' + ", ".join(set(
            [index_to_tag(slot)[2:] if index_to_tag(slot) != 'O' else 'O' for slot in slots]))
    return named_data


def get_data_per_intent():
    data_per_intent = {}
    for data in dataset['train']:
        intent = data['intent']
        if intent not in data_per_intent:
            data_per_intent[intent] = [data]
        else:
            data_per_intent[intent].append(data)
    return data_per_intent


def get_min_num_of_examples_per_intent():
    data_per_intent = get_data_per_intent()
    min_num_of_examples_per_intent = min(
        [len(data_per_intent[intent]) for intent in data_per_intent])
    return min_num_of_examples_per_intent


class T5GenerationFineTune(Dataset):
    def __init__(self, dataset, intent_slots, tokenizer, data_per_intent, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.intent_slots = intent_slots
        self.data_per_intent = data_per_intent

    def __len__(self):
        return len(self.dataset)

    # format of data (intent, slots_for_intent, example: text), all_examples_except_current
    def __getitem__(self, index):
        data = self.dataset[index]
        intent = data['intent']
        slots_for_intent = self.intent_slots[intent]
        text = data['text']
        all_examples_except_current = self.data_per_intent[intent][:index] + \
            self.data_per_intent[intent][index+1:]
        all_examples_except_current = [example['text']
                                       for example in all_examples_except_current]

        input_text = f"intent: {index_to_intent(intent)}\nslots: {slots_for_intent}\nexample: {text}"
        tokenized_text = self.tokenizer(
            input_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        tokenized_examples = [self.tokenizer(example, max_length=self.max_length, padding='max_length',
                                             truncation=True, return_tensors='pt') for example in all_examples_except_current]

        return tokenized_text.input_ids, tokenized_text.attention_mask, torch.Tensor(tokenized_examples)


class T5GenerationDataModule(LightningDataModule):
    def __init__(self, dataset, tokenizer, batch_size=8, max_length=512):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def setup(self, stage=None):
        self.intent_slots = get_intent_slots()
        self.data_per_intent = get_data_per_intent()
        self.train_dataset = T5GenerationFineTune(
            self.dataset['train'], self.intent_slots, self.tokenizer, self.data_per_intent, self.max_length)
        self.val_dataset = T5GenerationFineTune(
            self.dataset['validation'], self.intent_slots, self.tokenizer, self.data_per_intent, self.max_length)
        self.test_dataset = T5GenerationFineTune(
            self.dataset['test'], self.intent_slots, self.tokenizer, self.data_per_intent, self.max_length)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)
