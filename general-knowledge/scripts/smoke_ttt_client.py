import json
import zmq
from pathlib import Path


def main():
    data_path = Path('general-knowledge/data/synthetic_data/squad_val_synth_smoke.json')
    data = json.loads(data_path.read_text(encoding='utf-8'))
    item = data[0]

    msg = {
        'train_sequences': [f"{item['title']}\n{item['completions'][0]}"][:2],
        'eval_questions': item['questions'][:2],
        'lora_rank': 8,
        'lora_alpha': 16,
        'lora_dropout': 0.0,
        'finetune_epochs': 1,
        'finetune_lr': 5e-4,
        'batch_size': 1,
        'gradient_accumulation_steps': 1,
        'baseline_eval': True,
        'reward_mode': 'ttt',
    }

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect('tcp://127.0.0.1:5555')
    print('sending...')
    sock.send_json(msg)
    reply = sock.recv_json()
    print('reply keys:', list(reply.keys()))
    print('baseline_accuracy', reply.get('baseline_accuracy'))
    print('adapter_accuracy', reply.get('adapter_accuracy'))


if __name__ == '__main__':
    main()
