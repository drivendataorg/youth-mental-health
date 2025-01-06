import os, sys, logging
import re
from glob import glob
from tqdm import tqdm
import inspect
from itertools import combinations
import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from openai import OpenAI
import httpx

import util

PPT_GEN='''Rewrite the following text as simple as the original one. For example you can use synonyms, different syntax and change the order of narrative and so on.
Remember you need keep the same meaning/information as the original one.
original text:
{}
output:
'''

PPT_GEN2 = ''''You are an expert to generate law enforcement report and coroner/medical examiner report for suicide envent. The generated report will be used to train LLM models, so make sure the \
diversity of the generated report. Generated law enforcement report should put in <law></law> and coroner/medical examiner report put in <medical></medical>. Below are 3 examples of the report, \
please generate the 4th one, remember you only need generate the 4th example:
#####Example 1#####
<law>
{input_le1}
</law>
<medical>
{input_cme1}
</medical>

#####Example 2#####
<law>
{input_le2}
</law>
<medical>
{input_cme2}
</medical>

#####Example 3#####
<law>
{input_le3}
</law>
<medical>
{input_cme3}
</medical>

#####Example 4#####
'''

logger = logging.getLogger(__name__)


def gen_comb(l):
    exists, rsts = set(), []
    for c in combinations(l, 3):
        to_add = True
        ccs = list(combinations(c, 2))
        for cc in ccs:
            if cc in exists:
                to_add = False

        if to_add:
            yield c
            for cc in ccs:
                exists.add(cc)
    return rsts

def call_llama(llm, ppt, max_tokens, top_p=1, temperature=0, stop=[], **kwargs):
    try:
        msg = [
            {"role": "user", "content": ppt},
        ]
        output = llm.create_chat_completion(msg, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=stop)
        logger.debug("llamacpp output:%s", output)
        text= output["choices"][0]['message']['content']
    except Exception as e:
        logger.error(e)
        text = None
    return text

def get_vllm_model(modelid, gpu_memory_utilization=0.95, max_model_len=4096, quantization=None, enforce_eager=True, **kwargs):
    llm = LLM(model=modelid,
              quantization=quantization,
              dtype="half",
              enforce_eager=enforce_eager,
              gpu_memory_utilization=gpu_memory_utilization,
              max_model_len=max_model_len,
              trust_remote_code=True,
              **kwargs
              )
    tokenizer = llm.get_tokenizer()
    return llm, tokenizer


def call_vllm(llm, ppts, sp):
    outputs = llm.generate(ppts, sampling_params=sp, use_tqdm=True)
    outputs = sorted(outputs, key=lambda x: int(x.request_id))  # sort outputs by request_id
    #outputs = [llm.get_tokenizer().decode(output.prompt_token_ids) for output in outputs]
    logger.debug('llm output:%s', outputs)
    outputs = [[[o.text, o.finish_reason, o.stop_reason] for o in output.outputs] for output in outputs]
    return outputs

def call_vllm_service(client, model_name, ppts, temperature=0, top_p=1, max_tokens=2048, stop=[], timeout=None):
    completion = client.completions.create(model=model_name, prompt=ppts, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=stop, timeout=timeout)
    outputs = sorted(completion.choices, key=lambda x: int(x.index))
    outputs = [[[o.text, o.finish_reason, o.stop_reason]]for o in outputs]
    return outputs


def gen(args):
    os.makedirs(args.output_dir, exist_ok=True)
    df = util.load_data(args)
    exists_uids = [os.path.basename(fpath).split(".json")[0] for fpath in glob(f"{args.output_dir}/*.json")]
    df = df[~df.uid.isin(exists_uids)]
    if len(df)==0:
        logger.info('all uids already generated in %s', args.output_dir)
        return

    llm, tokenizer = get_vllm_model(args.modelid, cpu_offload_gb=args.cpu_offload_gb)
    logger.info('llm loaded')

    ppts_le, ppts_cme = [], []
    for le, cme in zip(df.NarrativeLE, df.NarrativeCME):
        ppts_le.append(tokenizer.apply_chat_template([ {"role": "user", "content":  PPT_GEN.format(le)} ], tokenize=False, add_generation_prompt=True))
        ppts_cme.append(tokenizer.apply_chat_template([ {"role": "user", "content":  PPT_GEN.format(cme)} ], tokenize=False, add_generation_prompt=True))

    sp = SamplingParams(temperature=1, top_p=1, max_tokens=1500, n=args.n_sample, seed=args.gen_seed)

    outputs_le = call_vllm(llm, ppts_le, sp)
    outputs_cme = call_vllm(llm, ppts_cme, sp)
    assert len(outputs_le) == len(outputs_cme)

    for uid, le, cme in zip(df.uid.values, outputs_le, outputs_cme):
        fpath = f"{args.output_dir}/{uid}.json"
        util.dump_json({uid: [le, cme]}, fpath)
    logger.info("Done")


def gen2(args):
    os.makedirs(args.output_dir, exist_ok=True)
    recs = util.load_data(args).to_records(index=False)
    exists_ids = [os.path.basename(fpath).split(".json")[0] for fpath in glob(f"{args.output_dir}/*/*.json")]
    exists_ids = set([tuple([int(id) for id in ids.split("_")])for ids in exists_ids])
    left_ids = []
    inds = np.arange(len(recs))
    rs = np.random.RandomState(3432)
    fpath = f"{args.data_dir}/combs.dump"
    if os.path.exists(fpath):
        combs = util.load_dump(fpath)
        logger.info('loaded combs from %s', fpath)
    else:
        combs = list(gen_comb(inds))
        rs.shuffle(combs)
        util.dump(combs, fpath)
        logger.info('combs saved to %s', fpath)
    for c in combs:
        if c not in exists_ids:
            left_ids.append(c)
            if len(left_ids)>=args.n_gen:
                break
    if len(left_ids)==0:
        logger.info('all generated')
        return
    # llm, tokenizer = get_vllm_model(args.modelid, cpu_offload_gb=args.cpu_offload_gb, max_model_len=8192, quantization=args.vll_quant, enforce_eager=not args.disable_eager)
    # NOTE - change
    llm, tokenizer = get_vllm_model(args.modelid, cpu_offload_gb=True, max_model_len=1024, quantization=args.vll_quant, enforce_eager=not args.disable_eager)
    logger.info('llm loaded')

    ppts = []
    for ids in left_ids:
        rec1, rec2, rec3 = [recs[id] for id in ids]
        ppt = PPT_GEN2.format(input_le1=rec1.NarrativeLE, input_cme1=rec1.NarrativeCME, input_le2=rec2.NarrativeLE, input_cme2=rec2.NarrativeCME,
                              input_le3=rec3.NarrativeLE, input_cme3=rec3.NarrativeCME)
        ppt = tokenizer.apply_chat_template([{"role": "user", "content": ppt}], tokenize=False, add_generation_prompt=True)
        ppts.append(ppt)

    sp = SamplingParams(temperature=0, top_p=1, max_tokens=2048, n=args.n_sample, stop=["</medical>"])
    logger.debug("ppts:%s", ppts)

    outputs = call_vllm(llm, ppts, sp)
    output_dir = f"{args.output_dir}/{args.prefix}"
    os.makedirs(output_dir, exist_ok=True)
    for output, ids in zip(outputs, left_ids):
        s_id = "_".join([str(id) for id in ids])
        fpath = f"{output_dir}/{s_id}.json"
        util.dump_json({s_id: output}, fpath)
        logger.debug('%s output:%s', s_id, output)

    logger.info("Done")
# NOTE - change
def gen3(args):
    # openai_api_key = "EMPTY"
    openai_api_key = "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"
    openai_api_base = "http://127.0.0.1:6666/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
        max_retries=0
    )
    timeout = httpx.Timeout(connect=10, read=12000000, write=12000000, pool=10000000)

    os.makedirs(args.output_dir, exist_ok=True)
    recs = util.load_data(args).to_records(index=False)
    all_fpaths = set(glob(f"../data/gen?/*/*.json"))
    exists_ids = [os.path.basename(fpath).split(".json")[0] for fpath in all_fpaths]
    exists_ids = set([tuple([int(id) for id in ids.split("_")])for ids in exists_ids])
    logger.info('existed:%s', len(exists_ids))
    left_ids = []
    inds = np.arange(len(recs))
    rs = np.random.RandomState(3432)
    fpath = f"{args.data_dir}/combs.dump"
    if os.path.exists(fpath):
        combs = util.load_dump(fpath)
        logger.info('loaded combs from %s', fpath)
    else:
        combs = list(gen_comb(inds))
        rs.shuffle(combs)
        util.dump(combs, fpath)
        logger.info('combs saved to %s', fpath)
    for c in combs:
        if c not in exists_ids:
            left_ids.append(c)
            if len(left_ids)>=args.n_gen:
                break
    if len(left_ids)==0:
        logger.info('all generated')
        return

    logger.info('left:%s', len(left_ids))

    tokenizer = AutoTokenizer.from_pretrained(args.modelid, trust_remote_code=args.trust_remote_code)
    ppts = []
    for ids in left_ids:
        rec1, rec2, rec3 = [recs[id] for id in ids]
        ppt = PPT_GEN2.format(input_le1=rec1.NarrativeLE, input_cme1=rec1.NarrativeCME, input_le2=rec2.NarrativeLE, input_cme2=rec2.NarrativeCME,
                              input_le3=rec3.NarrativeLE, input_cme3=rec3.NarrativeCME)
        ppt = tokenizer.apply_chat_template([{"role": "user", "content": ppt}], tokenize=False, add_generation_prompt=True)
        ppts.append(ppt)

    #sp = SamplingParams(temperature=0, top_p=1, max_tokens=2048, n=args.n_sample, stop=["</medical>"])
    #logger.debug("ppts:%s", ppts)

    outputs = call_vllm_service(client, args.prefix, ppts, temperature=args.temperature, top_p=1, max_tokens=2048, stop=["</medical>"], timeout=timeout)
    assert len(outputs) == len(left_ids)
    output_dir = f"{args.output_dir}/{args.prefix}"
    os.makedirs(output_dir, exist_ok=True)
    for output, ids in zip(outputs, left_ids):
        s_id = "_".join([str(id) for id in ids])
        fpath = f"{output_dir}/{s_id}.json"
        util.dump_json({s_id: output}, fpath)
        logger.debug('%s output:%s', s_id, output)

    logger.info("Done")





if __name__ == "__main__":
    args = util.parser.parse_args()
    gl = globals()
    if args.debug:
        util.set_logger(logging.DEBUG)
    else:
        util.set_logger()
    if args.method_name in gl:
        gl[args.method_name](args)
    else:
        logging.error('unknown method : %s', args.method_name)