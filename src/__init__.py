"""
src — Pacote principal do Hate-Speech Detector

Módulos:
    dataset   (Integrante B)  — Ingestão, limpeza, tokenização, DataLoaders e class weights.
    model     (Integrante C)  — Instanciação do Transformer pré-treinado e configuração de device.
    engine    (Integrante D/E)— Loop de treino, avaliação com métricas e salvamento de checkpoints.
    inference (Integrante A)  — Pipeline de inferência: carrega checkpoint e classifica textos novos.

Orquestrado por main.py (Integrante A).
"""
