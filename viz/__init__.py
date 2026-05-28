"""
viz/ — disease network representation storage.

Two artefact types per call:
  trajectories/  PNG curves of severity s(t) + symptom σ(t) per disease strategy
  embeddings/    PNG scatter (PCA of DistilBERT [CLS] embeddings) + raw .npz

Quick start — static trajectory atlas (no model needed):
    from viz.disease_viz import generate_disease_atlas
    paths = generate_disease_atlas("viz_output")

After FL training — save embedding snapshot:
    from viz.disease_viz import save_embedding_plot
    save_embedding_plot(events, fl_client.model, fl_client.lora_config,
                        output_dir="viz_output/embeddings", tag="round_05")
"""
