library(ggplot2)
library(ggrepel)

# Data from generate_figures_v2.R output (primary-action scoring - Jan 2026)
# Stats from: "--- By Model (ordered by mean score) ---"
# Note: Gemini 3 Pro has no valid readability data (n=0), using median (15.7) as placeholder
data <- data.frame(
  Model = c("GPT-5.2", "Grok 4.1", "Claude Sonnet 4", "Gemini 3 Pro",
            "Qwen 2.5 72B", "Llama 4", "GPT-4o"),
  Defensibility = c(0.78, 0.72, 0.66, 0.56, 0.45, 0.40, 0.39),
  Cost = c(2481, 1498, 765, 372, 437, 633, 645),
  Length = c(3335, 2101, 1396, 1462, 1757, 788, 852),
  Readability = c(18.0, 16.0, 17.6, 15.7, 15.1, 15.7, 15.5)  # Gemini uses median placeholder
)

# Conciseness = inverse of length (normalized for bubble size)
# Higher conciseness = smaller length = bigger bubble (more concise is better)
data$Conciseness <- max(data$Length) / data$Length

# Create bubble plot
p <- ggplot(data, aes(x = Cost, y = Defensibility)) +
  geom_point(aes(size = Conciseness, color = Readability), alpha = 0.8) +
  geom_text_repel(aes(label = Model),
                  size = 3.5,
                  box.padding = 0.5,
                  point.padding = 0.3,
                  segment.color = "gray50") +
  scale_color_gradient(low = "#2166AC", high = "#B2182B",
                       name = "Readability\n(higher = more complex)") +
  scale_size_continuous(name = "Conciseness\n(larger = more concise)",
                        range = c(4, 15)) +
  labs(x = "Mean Cost per Recommendation ($)",
       y = "Mean Defensibility Score (0-1)",
       title = "Cost vs Defensibility Trade-off (Primary-Action Scoring)",
       subtitle = "Bubble size = conciseness (inverse of response length); Color = readability complexity") +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 10, color = "gray40"),
    legend.position = "right",
    panel.grid.minor = element_blank()
  ) +
  scale_x_continuous(labels = scales::dollar_format()) +
  coord_cartesian(xlim = c(0, 3000))

ggsave("output/bubble_plot_cost_defensibility.pdf", p, width = 10, height = 7)
ggsave("output/bubble_plot_cost_defensibility.png", p, width = 10, height = 7, dpi = 300)

print("Plot saved to output/!")
print(p)
