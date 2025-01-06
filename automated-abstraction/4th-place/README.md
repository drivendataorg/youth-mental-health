# Solution - Youth Mental Health: Automated Abstraction
Team:bbb88

# Summary

I use 2 types of models SC(sequence classification transformers AutoModelForSequenceClassification) and AR(LLM Autoregressive AutoModelForCausalLM). 

For SC models I use backbone deberta v2 xl and xxl. Based on different input It has 2 sub model type SC1 and SC2. For each example it will predict logits for all variables. Binary cross entropy used for binary variables,
Cross entropy loss used for categorical variable.

- SC1 Concat NarrativeLE and NarrativeCME as model input.
- SC2 NarrativeLE and NarrativeCME  as seperated input sequence to get logits_LE and logits_CME, and the prediction logits is the max of them.


For AR I use backbones like llama 3.1 and so on. Both NarrativeLE and NarrativeCME was included as the context, for each variable will generated one prompt, 
since we have 23 variables, there will be 4000*23 training examples. The prmpts used refer ppts.py.

The training have 2 stage. Stage1 is for SFT(supervised finetuning). Stage2 I use LLM to generate SSL(semi supervised learning) data(refer gen_data.py)

I use some data augmentation like text rewrite(LLM), random order(NarrativeLE + NarrativeCMEand-> NarrativeCMEand + NarrativeLE) and regularization like rdrop for SC models.

The final submit was a weighted ensemble of the SSL mdoels(optuna was used to search the weight)

All the training details please refer yaa_train.ipynb


# Environment
Ubuntu 22.04, 1xRTX4090 24G, CPU 24 core 64G, python 3.10 

# install
- unzip competition data to data/yaa
- unzip data/gen.zip under folder data(by doing this you can skip section "gen rewrite" and "gen data" sub section under "gen semi" in the notebook)
- run ./setup.sh install requirements
- all program output will save under folder data
  
# gen rewrite text(EST 9Hours)
I use 6 LLMs to rewrite the text

Please refer "gen rewrite" section in the notebook, the output is under ../data/gen

# train stage1(EST 95Hours)
Totally 7x4=28 models(3 SC, 4 AR, 4fold)):SPLIT_debv2xl_d10 SFT_debv2xl_d120 SFT_debv2xxl_d36  AR_gemma2_2b_d02 AR_llama3d2_3b_d03 AR_llama3d2_1b_d37 AR_qw2d5_3b_d04

Please refer yaa_train.ipynb section "stage1" for train SFT models(KFIDS=0 means train fold 0), the model will save to f"../data/{MODELNAME}_KF{KFID}".

***The traing of SC(deberta) models is very unstable, with same parameters, it may divergence. If this happend I will detete the model and rerun with different pameters such like the seed(SEED variable in the notebook)
for mdoel weights initialization but keep the data seed(DATASEED variable in notebook) for fold split.
Generally speaking we will know if it converge well in about 4 epochs. I will take SFT_debv2xxl_d36 fold 3 for an example, after 4 epoch training, train loss is around 1.6, eval loss is around 1.4.***

Normal loss
```
{'loss': 5.9633, 'grad_norm': 19.5, 'learning_rate': 2.2857142857142858e-05, 'elapsed': 329.85977602005005, 'epoch': 0.3413333333333333}
{'loss': 4.2381, 'grad_norm': 6.65625, 'learning_rate': 4.5714285714285716e-05, 'elapsed': 686.0089826583862, 'epoch': 0.6826666666666666}
{'loss': 3.673, 'grad_norm': 10.6875, 'learning_rate': 4.9018867924528306e-05, 'elapsed': 1032.3829982280731, 'epoch': 1.024}
{'loss': 3.0366, 'grad_norm': 10.1875, 'learning_rate': 4.781132075471698e-05, 'elapsed': 1387.9873218536377, 'epoch': 1.3653333333333333}
{'loss': 2.6191, 'grad_norm': 72192.0, 'learning_rate': 4.660377358490566e-05, 'elapsed': 1750.4053580760956, 'epoch': 1.7066666666666666}
{'loss': 2.2829, 'grad_norm': 10.6875, 'learning_rate': 4.539622641509434e-05, 'elapsed': 2101.0117177963257, 'epoch': 2.048}
{'loss': 2.0201, 'grad_norm': 14.8125, 'learning_rate': 4.4188679245283023e-05, 'elapsed': 2455.564889192581, 'epoch': 2.389333333333333}
{'loss': 1.9321, 'grad_norm': 15.375, 'learning_rate': 4.29811320754717e-05, 'elapsed': 2816.5654780864716, 'epoch': 2.7306666666666666}
{'loss': 1.8926, 'grad_norm': 12.75, 'learning_rate': 4.177358490566038e-05, 'elapsed': 3166.080103635788, 'epoch': 3.072}
{'loss': 1.6972, 'grad_norm': 7.90625, 'learning_rate': 4.0566037735849064e-05, 'elapsed': 3517.670181751251, 'epoch': 3.413333333333333}
{'loss': 1.6803, 'grad_norm': 7.96875, 'learning_rate': 3.9358490566037735e-05, 'elapsed': 3862.1939096450806, 'epoch': 3.7546666666666666}
{'eval_loss': 1.4675954580307007, 'eval_runtime': 42.2632, 'eval_samples_per_second': 23.661, 'eval_steps_per_second': 11.831, 'elapsed': 4154.007340192795, 'epoch': 4.0}
{'loss': 1.6311, 'grad_norm': 12.9375, 'learning_rate': 3.815094339622642e-05, 'elapsed': 4247.438475608826, 'epoch': 4.096}
{'loss': 1.54, 'grad_norm': 8.625, 'learning_rate': 3.69433962264151e-05, 'elapsed': 4595.868144273758, 'epoch': 4.437333333333333}
{'loss': 1.5123, 'grad_norm': 8.75, 'learning_rate': 3.5735849056603775e-05, 'elapsed': 4945.429793834686, 'epoch': 4.778666666666666}
{'eval_loss': 1.3977057933807373, 'eval_runtime': 41.9078, 'eval_samples_per_second': 23.862, 'eval_steps_per_second': 11.931, 'elapsed': 5214.687683582306, 'epoch': 4.992}
{'loss': 1.4927, 'grad_norm': 9.0625, 'learning_rate': 3.452830188679245e-05, 'elapsed': 5342.696230173111, 'epoch': 5.12}
{'loss': 1.4166, 'grad_norm': 8.8125, 'learning_rate': 3.332075471698114e-05, 'elapsed': 5681.653786659241, 'epoch': 5.461333333333333}
{'loss': 1.3975, 'grad_norm': 100.0, 'learning_rate': 3.211320754716981e-05, 'elapsed': 6032.072592496872, 'epoch': 5.802666666666667}
{'eval_loss': 1.380799412727356, 'eval_runtime': 41.9007, 'eval_samples_per_second': 23.866, 'eval_steps_per_second': 11.933, 'elapsed': 6265.416762590408, 'epoch': 5.994666666666666}
{'loss': 1.3665, 'grad_norm': 9.6875, 'learning_rate': 3.090566037735849e-05, 'elapsed': 6410.55029129982, 'epoch': 6.144}
{'loss': 1.3217, 'grad_norm': 7.8125, 'learning_rate': 2.9698113207547174e-05, 'elapsed': 6781.753949403763, 'epoch': 6.485333333333333}
{'loss': 1.3518, 'grad_norm': 11.5, 'learning_rate': 2.8490566037735848e-05, 'elapsed': 7109.507408857346, 'epoch': 6.826666666666666}
{'eval_loss': 1.3804447650909424, 'eval_runtime': 41.891, 'eval_samples_per_second': 23.871, 'eval_steps_per_second': 11.936, 'elapsed': 7319.847545623779, 'epoch': 6.997333333333334}
{'loss': 1.3276, 'grad_norm': 8.8125, 'learning_rate': 2.7283018867924533e-05, 'elapsed': 7491.753851175308, 'epoch': 7.168}
{'loss': 1.2724, 'grad_norm': 7.59375, 'learning_rate': 2.6075471698113207e-05, 'elapsed': 7851.000901937485, 'epoch': 7.509333333333333}
{'loss': 1.2593, 'grad_norm': 10.5, 'learning_rate': 2.4867924528301888e-05, 'elapsed': 8188.468960762024, 'epoch': 7.850666666666667}
{'eval_loss': 1.373258352279663, 'eval_runtime': 42.0854, 'eval_samples_per_second': 23.761, 'eval_steps_per_second': 11.881, 'elapsed': 8384.206493139267, 'epoch': 8.0}
{'loss': 1.1868, 'grad_norm': 7.4375, 'learning_rate': 2.366037735849057e-05, 'elapsed': 8592.806854248047, 'epoch': 8.192}
{'loss': 1.2013, 'grad_norm': 6.375, 'learning_rate': 2.2452830188679247e-05, 'elapsed': 8932.14957332611, 'epoch': 8.533333333333333}
{'loss': 1.1964, 'grad_norm': 19.0, 'learning_rate': 2.1245283018867925e-05, 'elapsed': 9275.95428276062, 'epoch': 8.874666666666666}
{'eval_loss': 1.3619472980499268, 'eval_runtime': 42.0043, 'eval_samples_per_second': 23.807, 'eval_steps_per_second': 11.904, 'elapsed': 9445.416680335999, 'epoch': 8.992}
{'loss': 1.1614, 'grad_norm': 7.8125, 'learning_rate': 2.0037735849056606e-05, 'elapsed': 9674.98505616188, 'epoch': 9.216}
{'loss': 1.1984, 'grad_norm': 17.5, 'learning_rate': 1.8830188679245284e-05, 'elapsed': 10027.398931264877, 'epoch': 9.557333333333334}
{'loss': 1.1354, 'grad_norm': 8.75, 'learning_rate': 1.762264150943396e-05, 'elapsed': 10382.477838754654, 'epoch': 9.898666666666667}
{'eval_loss': 1.3726242780685425, 'eval_runtime': 42.02, 'eval_samples_per_second': 23.798, 'eval_steps_per_second': 11.899, 'elapsed': 10523.066487073898, 'epoch': 9.994666666666667}
{'loss': 1.1681, 'grad_norm': 6.75, 'learning_rate': 1.6415094339622643e-05, 'elapsed': 10761.310688495636, 'epoch': 10.24}
{'loss': 1.1026, 'grad_norm': 6.0, 'learning_rate': 1.520754716981132e-05, 'elapsed': 11105.295250415802, 'epoch': 10.581333333333333}
{'loss': 1.0816, 'grad_norm': 10.8125, 'learning_rate': 1.4000000000000001e-05, 'elapsed': 11470.924869060516, 'epoch': 10.922666666666666}
{'eval_loss': 1.364026665687561, 'eval_runtime': 41.9997, 'eval_samples_per_second': 23.81, 'eval_steps_per_second': 11.905, 'elapsed': 11599.314709186554, 'epoch': 10.997333333333334}
{'loss': 1.1015, 'grad_norm': 7.15625, 'learning_rate': 1.2792452830188681e-05, 'elapsed': 11858.970030784607, 'epoch': 11.264}
{'loss': 1.0867, 'grad_norm': 21.375, 'learning_rate': 1.1584905660377359e-05, 'elapsed': 12205.961586236954, 'epoch': 11.605333333333334}
{'loss': 1.1166, 'grad_norm': 7.78125, 'learning_rate': 1.0377358490566038e-05, 'elapsed': 12563.992066144943, 'epoch': 11.946666666666667}
{'eval_loss': 1.3647524118423462, 'eval_runtime': 42.1136, 'eval_samples_per_second': 23.745, 'eval_steps_per_second': 11.873, 'elapsed': 12655.403515338898, 'epoch': 12.0}
{'train_runtime': 12657.8585, 'train_samples_per_second': 3.555, 'train_steps_per_second': 0.11, 'train_loss': 1.7589152908325196, 'elapsed': 12658.198269605637, 'epoch': 12.0}
```

Abnormal loss
```
2024-10-20 03:14:08,083 - INFO - __main__ -   cls for trainer:<class 'trainer.Trainer'>
/home/tom/anaconda3/envs/yaa/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py:600: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.4 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
  return fn(*args, **kwargs)
/home/tom/anaconda3/envs/yaa/lib/python3.10/site-packages/torch/utils/checkpoint.py:295: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
  with torch.enable_grad(), device_autocast_ctx, torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):  # type: ignore[attr-defined]
{'loss': 6.5124, 'grad_norm': 53.5, 'learning_rate': 1.1428571428571429e-05, 'elapsed': 328.6619791984558, 'epoch': 0.3413333333333333}
{'loss': 4.6347, 'grad_norm': 13.3125, 'learning_rate': 2.2857142857142858e-05, 'elapsed': 683.8096528053284, 'epoch': 0.6826666666666666}
{'loss': 4.3661, 'grad_norm': 8.0625, 'learning_rate': 3.428571428571429e-05, 'elapsed': 1029.9332783222198, 'epoch': 1.024}
{'loss': 3.8753, 'grad_norm': 25.875, 'learning_rate': 4.5714285714285716e-05, 'elapsed': 1386.711844921112, 'epoch': 1.3653333333333333}
{'loss': 3.6069, 'grad_norm': 24.375, 'learning_rate': 4.9203187250996016e-05, 'elapsed': 1752.077222108841, 'epoch': 1.7066666666666666}
{'loss': 3.2298, 'grad_norm': 12.6875, 'learning_rate': 4.792828685258964e-05, 'elapsed': 2106.0061407089233, 'epoch': 2.048}
{'loss': 2.8288, 'grad_norm': 9.5625, 'learning_rate': 4.6653386454183266e-05, 'elapsed': 2465.4405806064606, 'epoch': 2.389333333333333}
{'loss': 2.652, 'grad_norm': 11.5625, 'learning_rate': 4.537848605577689e-05, 'elapsed': 2832.816324710846, 'epoch': 2.7306666666666666}
{'loss': 2.5843, 'grad_norm': 12.8125, 'learning_rate': 4.410358565737052e-05, 'elapsed': 3185.8573236465454, 'epoch': 3.072}
{'loss': 2.3453, 'grad_norm': 10.375, 'learning_rate': 4.2828685258964146e-05, 'elapsed': 3542.8296930789948, 'epoch': 3.413333333333333}
{'loss': 2.2699, 'grad_norm': 7.9375, 'learning_rate': 4.155378486055777e-05, 'elapsed': 3891.088043689728, 'epoch': 3.7546666666666666}
{'eval_loss': 2.1393468379974365, 'eval_runtime': 42.3699, 'eval_samples_per_second': 23.602, 'eval_steps_per_second': 11.801, 'elapsed': 4185.549817800522, 'epoch': 4.0}
{'loss': 2.3233, 'grad_norm': 8.5625, 'learning_rate': 4.0278884462151396e-05, 'elapsed': 4279.601939439774, 'epoch': 4.096}
{'loss': 2.2385, 'grad_norm': 9.0625, 'learning_rate': 3.900398406374502e-05, 'elapsed': 4633.739305734634, 'epoch': 4.437333333333333}
{'loss': 2.1784, 'grad_norm': 9.0, 'learning_rate': 3.772908366533865e-05, 'elapsed': 4987.976674795151, 'epoch': 4.778666666666666}
{'eval_loss': 2.0310287475585938, 'eval_runtime': 41.9176, 'eval_samples_per_second': 23.856, 'eval_steps_per_second': 11.928, 'elapsed': 5260.02686882019, 'epoch': 4.992}
{'loss': 2.2486, 'grad_norm': 31.75, 'learning_rate': 3.6454183266932277e-05, 'elapsed': 5389.80667090416, 'epoch': 5.12}
{'loss': 2.1836, 'grad_norm': 8.25, 'learning_rate': 3.51792828685259e-05, 'elapsed': 5732.276474952698, 'epoch': 5.461333333333333}
{'loss': 2.4345, 'grad_norm': 12.4375, 'learning_rate': 3.390438247011952e-05, 'elapsed': 6086.970051527023, 'epoch': 5.802666666666667}
{'eval_loss': 4.162196159362793, 'eval_runtime': 41.9013, 'eval_samples_per_second': 23.866, 'eval_steps_per_second': 11.933, 'elapsed': 6321.851492404938, 'epoch': 5.994666666666666}
{'loss': 4.3063, 'grad_norm': 4.65625, 'learning_rate': 3.2629482071713144e-05, 'elapsed': 6467.653731107712, 'epoch': 6.144}
{'loss': 4.2751, 'grad_norm': 4.5, 'learning_rate': 3.1354581673306775e-05, 'elapsed': 6845.36177277565, 'epoch': 6.485333333333333}
{'loss': 4.1994, 'grad_norm': 4.34375, 'learning_rate': 3.00796812749004e-05, 'elapsed': 7175.476159334183, 'epoch': 6.826666666666666}
{'eval_loss': 4.122344017028809, 'eval_runtime': 41.9642, 'eval_samples_per_second': 23.83, 'eval_steps_per_second': 11.915, 'elapsed': 7387.408467769623, 'epoch': 6.997333333333334}
{'loss': 4.2215, 'grad_norm': 4.375, 'learning_rate': 2.8804780876494025e-05, 'elapsed': 7561.138377666473, 'epoch': 7.168}
{'loss': 4.2351, 'grad_norm': 3.90625, 'learning_rate': 2.752988047808765e-05, 'elapsed': 7926.601666927338, 'epoch': 7.509333333333333}
{'loss': 4.202, 'grad_norm': 3.734375, 'learning_rate': 2.6254980079681274e-05, 'elapsed': 8267.272043943405, 'epoch': 7.850666666666667}
{'eval_loss': 4.11668062210083, 'eval_runtime': 42.1147, 'eval_samples_per_second': 23.745, 'eval_steps_per_second': 11.872, 'elapsed': 8465.5895113945, 'epoch': 8.0}
{'train_runtime': 8468.8317, 'train_samples_per_second': 5.314, 'train_steps_per_second': 0.165, 'train_loss': 3.4038465270996094, 'elapsed': 8469.239792108536, 'epoch': 8.0}
2024-10-20 05:35:17,515 - INFO - __main__ -   train DONE!
2024-10-20 05:35:17,515 - INFO - __main__ -   DONE!
```


# gen semi data(EST 125Hours)
I use LLM(3 shot) to generated data for SSL. Although I generated 260k examples, I only used ~50k of them because lack of computing resource. 

Please Refer "gen semi" section in the notebook for details. The outputs will be under ../data/gen[2-5]

# train stage2(EST 135Hours)
Totally 6x4=24 models(3 SC, 3 AR, 4 fold). Due to time limit, I only submit some fold of them, but all folds is used to search the ensemble weight.

Please refer section "stage2" in the notebook for train SSL models.

***Again training SC models may be unstable, I will give the normal and abnormal loss for SFT_debv2xl_semid11 fold 3***

Normal
```
  with torch.enable_grad(), device_autocast_ctx, torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):  # type: ignore[attr-defined]
{'loss': 6.0863, 'grad_norm': 18.0, 'learning_rate': 1.7021276595744682e-05, 'elapsed': 201.47030329704285, 'epoch': 0.17066666666666666}
{'loss': 4.1763, 'grad_norm': 6.28125, 'learning_rate': 3.4042553191489365e-05, 'elapsed': 405.40042328834534, 'epoch': 0.3413333333333333}
{'loss': 3.6411, 'grad_norm': 35.5, 'learning_rate': 4.9943693693693694e-05, 'elapsed': 628.4759414196014, 'epoch': 0.512}
{'loss': 2.7096, 'grad_norm': 10.1875, 'learning_rate': 4.9042792792792795e-05, 'elapsed': 838.3388595581055, 'epoch': 0.6826666666666666}
{'loss': 2.1326, 'grad_norm': 8.5, 'learning_rate': 4.8141891891891896e-05, 'elapsed': 1040.6308748722076, 'epoch': 0.8533333333333334}
{'loss': 1.8014, 'grad_norm': 14.5, 'learning_rate': 4.724099099099099e-05, 'elapsed': 1237.7058057785034, 'epoch': 1.024}
{'loss': 1.6702, 'grad_norm': 8.1875, 'learning_rate': 4.634009009009009e-05, 'elapsed': 1446.5583081245422, 'epoch': 1.1946666666666665}
{'loss': 1.5961, 'grad_norm': 7.28125, 'learning_rate': 4.543918918918919e-05, 'elapsed': 1656.6794328689575, 'epoch': 1.3653333333333333}
{'loss': 1.5552, 'grad_norm': 9.8125, 'learning_rate': 4.453828828828829e-05, 'elapsed': 1866.955719947815, 'epoch': 1.536}
{'loss': 1.4878, 'grad_norm': 6.125, 'learning_rate': 4.363738738738739e-05, 'elapsed': 2081.5937519073486, 'epoch': 1.7066666666666666}
{'loss': 1.4997, 'grad_norm': 10.1875, 'learning_rate': 4.273648648648649e-05, 'elapsed': 2278.886654615402, 'epoch': 1.8773333333333333}
{'eval_loss': 1.3127702474594116, 'eval_runtime': 20.8855, 'eval_samples_per_second': 47.88, 'eval_steps_per_second': 23.94, 'elapsed': 2439.085458755493, 'epoch': 2.0}
{'loss': 1.4138, 'grad_norm': 12.8125, 'learning_rate': 4.1835585585585585e-05, 'elapsed': 2495.618680715561, 'epoch': 2.048}
{'loss': 1.319, 'grad_norm': 7.21875, 'learning_rate': 4.0934684684684686e-05, 'elapsed': 2698.9394574165344, 'epoch': 2.2186666666666666}
{'loss': 1.3464, 'grad_norm': 9.25, 'learning_rate': 4.003378378378379e-05, 'elapsed': 2904.3952972888947, 'epoch': 2.389333333333333}
{'loss': 1.3382, 'grad_norm': 6.46875, 'learning_rate': 3.913288288288289e-05, 'elapsed': 3097.065095424652, 'epoch': 2.56}
{'loss': 1.2959, 'grad_norm': 6.03125, 'learning_rate': 3.823198198198198e-05, 'elapsed': 3305.8134031295776, 'epoch': 2.7306666666666666}
{'loss': 1.2942, 'grad_norm': 6.59375, 'learning_rate': 3.7331081081081084e-05, 'elapsed': 3523.5288667678833, 'epoch': 2.9013333333333335}
{'eval_loss': 1.2430545091629028, 'eval_runtime': 20.7854, 'eval_samples_per_second': 48.111, 'eval_steps_per_second': 24.055, 'elapsed': 3666.951335668564, 'epoch': 2.997333333333333}
{'loss': 1.2643, 'grad_norm': 4.625, 'learning_rate': 3.6430180180180185e-05, 'elapsed': 3769.925977230072, 'epoch': 3.072}
{'loss': 1.2057, 'grad_norm': 5.71875, 'learning_rate': 3.552927927927928e-05, 'elapsed': 3994.3013274669647, 'epoch': 3.2426666666666666}
{'loss': 1.2271, 'grad_norm': 8.0, 'learning_rate': 3.462837837837838e-05, 'elapsed': 4214.875334024429, 'epoch': 3.413333333333333}
{'loss': 1.1829, 'grad_norm': 5.40625, 'learning_rate': 3.372747747747748e-05, 'elapsed': 4415.976058959961, 'epoch': 3.584}
{'loss': 1.2048, 'grad_norm': 6.84375, 'learning_rate': 3.282657657657658e-05, 'elapsed': 4622.804242610931, 'epoch': 3.7546666666666666}
{'loss': 1.1914, 'grad_norm': 6.125, 'learning_rate': 3.192567567567568e-05, 'elapsed': 4829.530074357986, 'epoch': 3.9253333333333336}
{'eval_loss': 1.2225100994110107, 'eval_runtime': 20.8013, 'eval_samples_per_second': 48.074, 'eval_steps_per_second': 24.037, 'elapsed': 4936.480255365372, 'epoch': 4.0}
{'loss': 1.1563, 'grad_norm': 5.125, 'learning_rate': 3.102477477477478e-05, 'elapsed': 5050.677282333374, 'epoch': 4.096}
{'loss': 1.1077, 'grad_norm': 5.28125, 'learning_rate': 3.0123873873873877e-05, 'elapsed': 5259.623512744904, 'epoch': 4.266666666666667}
{'loss': 1.1522, 'grad_norm': 7.28125, 'learning_rate': 2.9222972972972972e-05, 'elapsed': 5469.5643763542175, 'epoch': 4.437333333333333}
{'loss': 1.1506, 'grad_norm': 10.25, 'learning_rate': 2.8322072072072076e-05, 'elapsed': 5684.857266426086, 'epoch': 4.608}
{'loss': 1.1298, 'grad_norm': 6.6875, 'learning_rate': 2.7421171171171174e-05, 'elapsed': 5879.633967638016, 'epoch': 4.778666666666666}
{'loss': 1.1298, 'grad_norm': 6.0, 'learning_rate': 2.652027027027027e-05, 'elapsed': 6089.0006222724915, 'epoch': 4.949333333333334}
{'eval_loss': 1.1912381649017334, 'eval_runtime': 20.7851, 'eval_samples_per_second': 48.112, 'eval_steps_per_second': 24.056, 'elapsed': 6186.205631017685, 'epoch': 4.997333333333334}
{'loss': 1.1125, 'grad_norm': 13.9375, 'learning_rate': 2.5619369369369373e-05, 'elapsed': 6332.302077770233, 'epoch': 5.12}
{'loss': 1.1119, 'grad_norm': 5.5, 'learning_rate': 2.471846846846847e-05, 'elapsed': 6527.83264040947, 'epoch': 5.290666666666667}
{'loss': 1.0976, 'grad_norm': 4.96875, 'learning_rate': 2.381756756756757e-05, 'elapsed': 6741.39097571373, 'epoch': 5.461333333333333}
{'loss': 1.0697, 'grad_norm': 5.65625, 'learning_rate': 2.2916666666666667e-05, 'elapsed': 6958.956504821777, 'epoch': 5.632}
{'loss': 1.0829, 'grad_norm': 5.46875, 'learning_rate': 2.2015765765765768e-05, 'elapsed': 7174.375030040741, 'epoch': 5.802666666666667}
{'loss': 1.0906, 'grad_norm': 5.3125, 'learning_rate': 2.1114864864864866e-05, 'elapsed': 7381.28334736824, 'epoch': 5.973333333333334}
{'eval_loss': 1.1919764280319214, 'eval_runtime': 20.8085, 'eval_samples_per_second': 48.057, 'eval_steps_per_second': 24.029, 'elapsed': 7434.167377710342, 'epoch': 6.0}
{'loss': 1.0575, 'grad_norm': 5.53125, 'learning_rate': 2.0213963963963964e-05, 'elapsed': 7598.236744642258, 'epoch': 6.144}
{'loss': 1.0756, 'grad_norm': 5.125, 'learning_rate': 1.9313063063063065e-05, 'elapsed': 7808.711550474167, 'epoch': 6.314666666666667}
{'loss': 1.0397, 'grad_norm': 5.84375, 'learning_rate': 1.8412162162162163e-05, 'elapsed': 8021.415939807892, 'epoch': 6.485333333333333}
{'loss': 1.0389, 'grad_norm': 4.875, 'learning_rate': 1.751126126126126e-05, 'elapsed': 8245.792765378952, 'epoch': 6.656}
{'loss': 1.059, 'grad_norm': 5.09375, 'learning_rate': 1.6610360360360362e-05, 'elapsed': 8435.931077480316, 'epoch': 6.826666666666666}
{'loss': 1.0328, 'grad_norm': 5.4375, 'learning_rate': 1.570945945945946e-05, 'elapsed': 8633.518298387527, 'epoch': 6.997333333333334}
{'eval_loss': 1.1930232048034668, 'eval_runtime': 20.8037, 'eval_samples_per_second': 48.068, 'eval_steps_per_second': 24.034, 'elapsed': 8656.865202903748, 'epoch': 6.997333333333334}
{'loss': 1.0432, 'grad_norm': 5.03125, 'learning_rate': 1.4808558558558558e-05, 'elapsed': 8867.687192440033, 'epoch': 7.168}
{'loss': 1.0467, 'grad_norm': 5.125, 'learning_rate': 1.3907657657657657e-05, 'elapsed': 9086.794427633286, 'epoch': 7.338666666666667}
{'loss': 1.0696, 'grad_norm': 6.46875, 'learning_rate': 1.3006756756756757e-05, 'elapsed': 9291.839955568314, 'epoch': 7.509333333333333}
{'loss': 1.0211, 'grad_norm': 6.15625, 'learning_rate': 1.2105855855855857e-05, 'elapsed': 9494.812783002853, 'epoch': 7.68}
{'loss': 0.9959, 'grad_norm': 4.9375, 'learning_rate': 1.1204954954954954e-05, 'elapsed': 9696.790886163712, 'epoch': 7.850666666666667}
{'eval_loss': 1.190372109413147, 'eval_runtime': 20.8037, 'eval_samples_per_second': 48.068, 'eval_steps_per_second': 24.034, 'elapsed': 9900.43697309494, 'epoch': 8.0}
{'loss': 1.0282, 'grad_norm': 5.375, 'learning_rate': 1.0304054054054054e-05, 'elapsed': 9927.385549545288, 'epoch': 8.021333333333333}
{'loss': 1.0508, 'grad_norm': 6.84375, 'learning_rate': 9.403153153153154e-06, 'elapsed': 10136.577413797379, 'epoch': 8.192}
{'loss': 0.9985, 'grad_norm': 5.3125, 'learning_rate': 8.502252252252253e-06, 'elapsed': 10359.068918466568, 'epoch': 8.362666666666666}
{'loss': 1.0043, 'grad_norm': 4.3125, 'learning_rate': 7.601351351351352e-06, 'elapsed': 10561.236629486084, 'epoch': 8.533333333333333}
{'loss': 1.0509, 'grad_norm': 4.71875, 'learning_rate': 6.7004504504504505e-06, 'elapsed': 10765.152240276337, 'epoch': 8.704}
{'loss': 1.027, 'grad_norm': 4.5, 'learning_rate': 5.799549549549549e-06, 'elapsed': 10964.099436044693, 'epoch': 8.874666666666666}
{'eval_loss': 1.1882485151290894, 'eval_runtime': 20.7865, 'eval_samples_per_second': 48.108, 'eval_steps_per_second': 24.054, 'elapsed': 11136.985196590424, 'epoch': 8.997333333333334}
{'loss': 1.0455, 'grad_norm': 5.75, 'learning_rate': 4.898648648648649e-06, 'elapsed': 11205.632209539413, 'epoch': 9.045333333333334}
{'loss': 1.0041, 'grad_norm': 4.3125, 'learning_rate': 3.997747747747748e-06, 'elapsed': 11414.230979681015, 'epoch': 9.216}
{'loss': 1.0482, 'grad_norm': 6.0625, 'learning_rate': 3.096846846846847e-06, 'elapsed': 11631.692245960236, 'epoch': 9.386666666666667}
{'loss': 1.023, 'grad_norm': 4.53125, 'learning_rate': 2.195945945945946e-06, 'elapsed': 11828.369939088821, 'epoch': 9.557333333333334}
{'loss': 1.0002, 'grad_norm': 5.625, 'learning_rate': 1.2950450450450451e-06, 'elapsed': 12025.538802862167, 'epoch': 9.728}
{'loss': 1.0054, 'grad_norm': 5.78125, 'learning_rate': 3.941441441441441e-07, 'elapsed': 12227.841891288757, 'epoch': 9.898666666666667}
{'eval_loss': 1.1899361610412598, 'eval_runtime': 20.8122, 'eval_samples_per_second': 48.049, 'eval_steps_per_second': 24.024, 'elapsed': 12340.033514738083, 'epoch': 9.973333333333333}
{'train_runtime': 12349.3461, 'train_samples_per_second': 4.859, 'train_steps_per_second': 0.151, 'train_loss': 1.3902504977057961, 'elapsed': 12349.669184684753, 'epoch': 9.973333333333333}
2024-11-17 14:14:02,975 - INFO - __main__ -   train DONE!
2024-11-17 14:14:02,975 - INFO - __main__ -   DONE!
```
Abnormal
```
  with torch.enable_grad(), device_autocast_ctx, torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):  # type: ignore[attr-defined]
{'loss': 6.2626, 'grad_norm': 17.375, 'learning_rate': 1.7021276595744682e-05, 'elapsed': 220.968510389328, 'epoch': 0.17066666666666666}
{'loss': 4.2579, 'grad_norm': 7.9375, 'learning_rate': 3.4042553191489365e-05, 'elapsed': 433.06526350975037, 'epoch': 0.3413333333333333}
{'loss': 3.5043, 'grad_norm': 14.6875, 'learning_rate': 4.9943693693693694e-05, 'elapsed': 665.8582072257996, 'epoch': 0.512}
{'loss': 2.7245, 'grad_norm': 26.5, 'learning_rate': 4.9042792792792795e-05, 'elapsed': 851.6287431716919, 'epoch': 0.6826666666666666}
{'loss': 2.2649, 'grad_norm': 37.0, 'learning_rate': 4.8141891891891896e-05, 'elapsed': 1064.5378336906433, 'epoch': 0.8533333333333334}
{'loss': 2.1093, 'grad_norm': 31.75, 'learning_rate': 4.724099099099099e-05, 'elapsed': 1270.45148396492, 'epoch': 1.024}
{'loss': 4.3886, 'grad_norm': 12.3125, 'learning_rate': 4.634009009009009e-05, 'elapsed': 1480.802565574646, 'epoch': 1.1946666666666665}
{'loss': 4.2553, 'grad_norm': 17.125, 'learning_rate': 4.543918918918919e-05, 'elapsed': 1688.7342157363892, 'epoch': 1.3653333333333333}
{'loss': 4.1768, 'grad_norm': 7.25, 'learning_rate': 4.453828828828829e-05, 'elapsed': 1888.1512727737427, 'epoch': 1.536}
{'loss': 4.2151, 'grad_norm': 8.6875, 'learning_rate': 4.363738738738739e-05, 'elapsed': 2102.080956220627, 'epoch': 1.7066666666666666}
{'loss': 4.2234, 'grad_norm': 9.0, 'learning_rate': 4.273648648648649e-05, 'elapsed': 2306.511698484421, 'epoch': 1.8773333333333333}
{'eval_loss': 4.172544479370117, 'eval_runtime': 20.8599, 'eval_samples_per_second': 47.939, 'eval_steps_per_second': 23.969, 'elapsed': 2487.169590473175, 'epoch': 2.0}
```

# submission for inference
- under fold data/hfmodels/unsloth, create soft links to the transformoers download model weights, change "/mnt/usbdisk1/data/cache/huggingface/hub/" to your cache dir(or copy it)
```
  
ls -ltrh unsloth/
total 24K
lrwxrwxrwx 1 tom tom 130 10月 30 11:53 Llama-3.2-1B-Instruct -> /mnt/usbdisk1/data/cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/snapshots/50ea995812f20bf680a17a02cfbc4f90ff4d9c0e
lrwxrwxrwx 1 tom tom 130 11月  2 07:23 Llama-3.2-3B-Instruct -> /mnt/usbdisk1/data/cache/huggingface/hub/models--unsloth--Llama-3.2-3B-Instruct/snapshots/bc836a93eabc97432d7be9faedddf045ca7ad8fc
lrwxrwxrwx 1 tom tom 122 11月  3 21:06 gemma-2-2b-it -> /mnt/usbdisk1/data/cache/huggingface/hub/models--unsloth--gemma-2-2b-it/snapshots/457f2e15bf550c227ce6ad86e2ec108d3e42c106
lrwxrwxrwx 1 tom tom 128 11月  6 07:27 Qwen2.5-3B-Instruct -> /mnt/usbdisk1/data/cache/huggingface/hub/models--unsloth--Qwen2.5-3B-Instruct/snapshots/bfc139e73f57ef880e717a91b0fa74dd9f0f97ae

```
- run below command will generate the submission.zip under folder ../data/submission
```
./gen_submit.sh "AR_gemma2_2b_semid01_KF0 AR_gemma2_2b_semid01_KF1  AR_llama3d2_3b_semid01_KF0 AR_llama3d2_3b_semid01_KF1 AR_llama3d2_3b_semid01_KF2 AR_llama3d2_1b_semid05_KF0 AR_llama3d2_1b_semid05_KF1 SPLIT_debv2xl_semid06_KF0 SPLIT_debv2xl_semid06_KF1 SFT_debv2xl_semid11_KF0 SFT_debv2xl_semid11_KF1 SFT_debv2xxl_semid03_KF0 SFT_debv2xxl_semid03_KF1" 0.05
```

