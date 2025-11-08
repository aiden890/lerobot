PI05Config 에 LoRA 설정 필드 추가: use_vision_lora, use_language_lora, lora_alpha, lora_dropout, lora_target_modules 등으로 어떤 백본에 LoRA를 적용할지 CLI에서 제어할 수 있게 만듭니다.
PI05 모델 초기화 시 Paligemma 비전·언어 모듈(및 필요 시 expert projector)을 peft.LoraConfig + get_peft_model 로 감싸고, 기본 가중치를 freeze한 뒤 LoRA 파라미터만 학습하도록 구성합니다.
    PI05 모델에는 Paligemma 기반 비전·언어 백본과 action expert(head)가 포함되는데, 각 모듈을 LoRA로 감싸면 다음 순서로 진행됩니다:
    LoRA 구성: LoraConfig(r=64, lora_alpha=32, lora_dropout=0.05, target_modules=[...])처럼 랭크·스케일·적용할 레이어 이름을 설정합니다.
    모듈 래핑: get_peft_model(paligemma.language_model, lora_config)을 호출하면 해당 모듈에 LoRA 어댑터가 삽입된 객체가 반환됩니다. 동일하게 비전 타워나 action expert에도 각각의 설정으로 호출합니다.
    파라미터 freeze: Base 모델 가중치에 requires_grad=False를 설정하고, LoRA 어댑터 파라미터만 학습대상으로 남겨둡니다. 이렇게 하면 forward는 기존 가중치+LoRA 델타가 합쳐져 실행되고, backward/update는 어댑터 파라미터에만 적용됩니다.
    훈련/저장: 학습 시에는 LoRA 파라미터만 업데이트되므로 자원 소모가 작고, 저장할 때도 peft_model.save_pretrained()로 어댑터만 별도 디렉터리에 보관할 수 있습니다. 추론 시에는 base 모델을 로드한 뒤 동일한 LoRA 어댑터를 from_pretrained로 붙이면 동일한 효과를 얻습니다.
Optimizer 파라미터 그룹이나 학습률을 LoRA 전용으로 조정하고, 학습/추론 파이프라인에서 LoRA 어댑터 저장·로딩을 지원해 실험과 배포를 쉽게 합니다.