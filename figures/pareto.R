library(ggplot2)
library(ggrepel)

# Data from generate_figures_v2.R output (primary-action scoring - Feb 2026)
llm_summary <- data.frame(
  llm_display = c("GPT-5.2", "GPT-5.2 Concise", "Grok 4.1", "Claude Sonnet 4", "DeepSeek R1",
                   "Mistral Large", "Gemini 3 Pro", "Claude 3.5 Haiku",
                   "Qwen 2.5 72B", "Gemini 2.0 Flash", "Llama 4 Scout",
                   "Llama 4 Maverick", "GPT-4o", "GPT-4o Mini"),
  mean_score = c(0.72, 0.56, 0.67, 0.61, 0.59, 0.54, 0.53, 0.43,
                 0.42, 0.41, 0.39, 0.37, 0.36, 0.31),
  mean_cost = c(3047, 2459, 1975, 1098, 1525, 1054, 1100, 1011,
                612, 828, 859, 750, 739, 397)
)

# Color palette - consistent with other figures
llm_colors <- c(
  "GPT-5.2" = "#1f77b4",
  "GPT-5.2 Concise" = "#aec7e8",
  "GPT-4o" = "#e377c2",
  "GPT-4o Mini" = "#f7b6d2",
  "Claude Sonnet 4" = "#2ca02c",
  "Claude 3.5 Haiku" = "#98df8a",
  "Grok 4.1" = "#ff7f0e",
  "Gemini 3 Pro" = "#8c564b",
  "Gemini 2.0 Flash" = "#c49c94",
  "Llama 4 Maverick" = "#d62728",
  "Llama 4 Scout" = "#ff9896",
  "Qwen 2.5 72B" = "#9467bd",
  "DeepSeek R1" = "#17becf",
  "Mistral Large" = "#bcbd22"
)

# Identify Pareto-optimal models (maximize defensibility, minimize cost)
pareto_models <- llm_summary[order(llm_summary$mean_cost), ]
pareto_models$is_pareto <- {
  max_def_so_far <- cummax(pareto_models$mean_score)
  pareto_models$mean_score >= max_def_so_far
}
pareto_models <- pareto_models[pareto_models$is_pareto, ]

# Define zone boundaries
x_max <- 3500
y_max <- 1.0

# Clinical Efficiency Frontier zone (top-left: high defensibility, low cost)
cef_xmin <- 0; cef_xmax <- 1200; cef_ymin <- 0.55; cef_ymax <- y_max

# Algorithmic Defensive Medicine zone (top-right: high defensibility, high cost)
adm_xmin <- 1800; adm_xmax <- x_max; adm_ymin <- 0.55; adm_ymax <- y_max

# Build the plot
p <- ggplot(llm_summary, aes(x = mean_cost, y = mean_score)) +
  # Shaded zones
  annotate("rect", xmin = cef_xmin, xmax = cef_xmax, ymin = cef_ymin, ymax = cef_ymax,
           fill = "#2ecc71", alpha = 0.12) +
  annotate("rect", xmin = adm_xmin, xmax = adm_xmax, ymin = adm_ymin, ymax = adm_ymax,
           fill = "#e74c3c", alpha = 0.10) +
  # Zone labels
  annotate("text", x = (cef_xmin + cef_xmax) / 2, y = cef_ymax - 0.02,
           label = "Clinical Efficiency\nFrontier", fontface = "italic",
           size = 3.5, color = "#1a8a4a", vjust = 1) +
  annotate("text", x = (adm_xmin + adm_xmax) / 2, y = adm_ymax - 0.02,
           label = "Algorithmic\nDefensive Medicine", fontface = "italic",
           size = 3.5, color = "#c0392b", vjust = 1) +
  # Pareto frontier line
  geom_line(data = pareto_models[order(pareto_models$mean_cost), ],
            aes(x = mean_cost, y = mean_score),
            linetype = "dashed", color = "gray40", linewidth = 0.8) +
  # Optimization gap arrow (from GPT-5.2 toward the efficiency frontier)
  annotate("segment",
           x = 3047, y = 0.72,
           xend = 1100, yend = 0.72,
           arrow = arrow(length = unit(0.25, "cm"), ends = "last", type = "closed"),
           color = "gray30", linewidth = 0.6, linetype = "dotted") +
  annotate("text", x = 2050, y = 0.745,
           label = "Optimization Gap", fontface = "italic",
           size = 3.2, color = "gray30") +
  # Model points
  geom_point(aes(color = llm_display), size = 5, alpha = 0.85) +
  geom_text_repel(aes(label = llm_display), size = 3.8, box.padding = 0.5,
                  point.padding = 0.3, max.overlaps = 10, seed = 42,
                  segment.color = "gray50",
                  bg.color = "white", bg.r = 0.15) +
  # Pareto frontier label
  annotate("text", x = max(llm_summary$mean_cost) * 0.85, y = 0.35,
           label = "Pareto Frontier", fontface = "italic", size = 3.2, color = "gray40") +
  # Scales and labels
  scale_color_manual(values = llm_colors, guide = "none") +
  scale_x_continuous(labels = scales::dollar_format()) +
  scale_y_continuous(breaks = seq(0.3, 0.8, 0.1)) +
  coord_cartesian(xlim = c(0, x_max), ylim = c(0.25, 0.85)) +
  labs(
    x = "Mean Cost per Recommendation ($)",
    y = "Mean Defensibility Score (0-1)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    plot.margin = margin(10, 15, 10, 10)
  )

# Save
dir.create("figures_v2", showWarnings = FALSE)
ggsave("output/fig3_pareto_standalone.pdf", p, width = 8, height = 6)
ggsave("output/fig3_pareto_standalone.png", p, width = 8, height = 6, dpi = 300)

cat("Standalone Pareto figure saved to output/fig3_pareto_standalone.pdf\n")
print(p)
