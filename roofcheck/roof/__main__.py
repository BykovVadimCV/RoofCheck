"""python -m roofcheck.roof sample.json --output out/example --seed 42"""
import argparse
import json
from pathlib import Path
from . import RoofSample, GenerationConfig, generate, save_result


def main():
    parser = argparse.ArgumentParser(description='Generate one roof snow sample with 0/1/255 labels')
    parser.add_argument('sample',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=0)
    args = parser.parse_args()
    payload = json.loads(args.sample.read_text())
    sample = payload['sample']
    for key in ('image_path','roof_mask','snow_mask','inspection_roi','occluder_mask'):
        if sample.get(key):
            sample[key] = args.sample.resolve().parent/sample[key]
    spec = RoofSample(**sample)
    generation = payload.get('generation',{})
    if generation.get('texture_path'):
        generation['texture_path'] = str(args.sample.resolve().parent / generation['texture_path'])
    result = generate(spec,GenerationConfig(**generation),seed=args.seed)
    save_result(spec,result,args.output)
    print(json.dumps(result.metadata['coverage'],indent=2))


if __name__ == '__main__':
    main()
