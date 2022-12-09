import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration, T5Tokenizer
from undecorated import undecorated
from types import MethodType


class T5DataGenerator(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        print(args)
        self.args = args
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.args['model_name'])
        self.tokenizer = T5Tokenizer.from_pretrained(
            self.args['tokenizer_name'])

        # allow grad in generate
        generate_with_grad = undecorated(self.model.generate)
        self.model.generate_with_grad = MethodType(
            generate_with_grad, self.model)

    # have the model generate multiple unique candidate outputs for a given input

    def forward(self, input_ids, attention_mask, num_return_sequences):
        return self.model.generate_with_grad(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            max_length=50,
            top_k=self.args['top_k'],
            top_p=0.95,
            num_return_sequences=self.args['num_return_sequences'],
        )

    # for a given training step, generate a batch of candidate outputs for a given input
    # input_ids are the tokenized examples of the data we want to generate
    # the labels are all of the potential outputs the model could generate for a given intent
    # model out shape (bs * num_return_sequences, max_seq_len)
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids=input_ids, attention_mask=attention_mask,
                       num_return_sequences=self.args['num_return_sequences'])
        # for each output sequence, calculate the least loss relative to all of teh examples in labels

        self.log('train_loss', loss)

        loss = outputs[0]
        return loss


args = {
    'model_name': 't5-small',
    'tokenizer_name': 't5-small',
    'top_k': 5,
    'num_return_sequences': 5,
}

test_model = T5DataGenerator(args)

inp = ['this is a test input', 'this is a second test input']
tokenizer = T5Tokenizer.from_pretrained('t5-small')
input_ids = tokenizer(inp, return_tensors='pt', padding=True,
                      truncation=True, max_length=512)['input_ids']

s = test_model(
    input_ids, None, 1)
print(s.shape)
print(tokenizer.batch_decode(s, skip_special_tokens=True))

# x = test_model(input_ids, 1)
# print(x)
# print(x.shape)
# print(tokenizer.batch_decode(x))

# model.save(model.state_dict(), )
