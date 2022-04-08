from motion import MotionGenerator
from orientation import OrientationGenerator
from distance import DistanceGenerator
from metaphor import MetaphorGenerator
from sklearn.metrics import accuracy_score as accuracy

from transformers import T5Tokenizer, T5ForConditionalGeneration


def evaluate(gen):
    # adapted from https://huggingface.co/docs/transformers/model_doc/t5#training
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    count = 0
    total = 0

    for batch_x, batch_y in gen.batch(
        batch_size=32,
    ):
        transform = lambda x: f"mnli hypothesis: {x[1]} premise: {x[0]}" # task setup for T5
        batch_x_transformed = list(map(transform, batch_x))
        
        inputs = tokenizer(batch_x_transformed, return_tensors="pt", padding=True)
        output_sequences = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            do_sample=False,
        )
        answers = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

        acc = accuracy(answers, batch_y)
        count += acc * len(answers)
        total += len(answers)

        for (p, h), e, a in zip(batch_x, batch_y, answers):
            print(f"{e} {a}: {p} => {h}")
        break
        
    print('Acc:', count / total)


if __name__ == '__main__':
    
    motion_gen = MotionGenerator()
    #evaluate(motion_gen)
    
    orientation_gen = OrientationGenerator()
    #evaluate(orientation_gen)
    
    distance_gen = DistanceGenerator(sample=0.01)
    #evaluate(distance_gen)
    
    metaphor_gen = MetaphorGenerator()
    evaluate(metaphor_gen)