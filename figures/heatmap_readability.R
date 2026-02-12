library(ggplot2)
library(reshape2)
library(dplyr)

# Load the actual data
runs <- read.csv("../results/runs.csv")

# Filter to valid scores only
runs <- runs %>%
  filter(score_valid == TRUE | score_valid == "True") %>%
  filter(!is.na(score_0_2))

# Convert score to 0-1 scale
runs$defensibility <- runs$score_0_2 / 2

# Readability metrics to analyze
readability_cols <- c("transformer_readability_score", "semantic_coherence_local",
                      "flesch_kincaid_grade", "semantic_coherence_global",
                      "smog_index", "lexical_overlap_global",
                      "lexical_overlap_adjacent", "pronoun_density")

# Pretty names for metrics
metric_names <- c("Transformer\nReadability", "Semantic Coherence\n(Local)",
                  "Flesch-Kincaid\nGrade", "Semantic Coherence\n(Global)",
                  "SMOG Index", "Lexical Overlap\n(Global)",
                  "Lexical Overlap\n(Adjacent)", "Pronoun\nDensity")

# Compute correlations by model
models <- unique(runs$llm_name)
results <- data.frame()

for (model in models) {
  model_data <- runs %>% filter(llm_name == model)

  for (i in seq_along(readability_cols)) {
    col <- readability_cols[i]
    if (col %in% names(model_data)) {
      # Remove NAs for this specific metric
      valid_data <- model_data[!is.na(model_data[[col]]) & !is.na(model_data$defensibility), ]
      if (nrow(valid_data) > 10) {
        cor_val <- cor(valid_data$defensibility, valid_data[[col]], method = "pearson")
        results <- rbind(results, data.frame(
          Model = model,
          Metric = metric_names[i],
          Correlation = cor_val
        ))
      }
    }
  }
}

# Order models by mean defensibility
model_order <- runs %>%
  group_by(llm_name) %>%
  summarise(mean_def = mean(defensibility, na.rm = TRUE)) %>%
  arrange(desc(mean_def)) %>%
  pull(llm_name)

results$Model <- factor(results$Model, levels = rev(model_order))
results$Metric <- factor(results$Metric, levels = metric_names)

# Create heatmap
p <- ggplot(results, aes(x = Metric, y = Model, fill = Correlation)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.2f", Correlation)),
            color = ifelse(abs(results$Correlation) > 0.25, "white", "black"),
            size = 3.5) +
  scale_fill_gradient2(low = "#2166AC", mid = "white", high = "#B2182B",
                       midpoint = 0, limits = c(-0.5, 0.5),
                       name = "Correlation\nwith\nDefensibility") +
  labs(title = "Readability Metrics vs Defensibility by Model",
       subtitle = "Pearson correlation of each readability metric with defensibility score (per-consultation level)",
       x = "", y = "") +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 10, color = "gray40"),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
    axis.text.y = element_text(size = 11),
    panel.grid = element_blank(),
    legend.position = "right"
  )

ggsave("output/heatmap_readability_by_model.pdf", p, width = 12, height = 6)
ggsave("output/heatmap_readability_by_model.png", p, width = 12, height = 6, dpi = 300)

print("Heatmap saved!")
print(p)

# Print summary
cat("\nCorrelations by model:\n")
print(dcast(results, Model ~ Metric, value.var = "Correlation"))
