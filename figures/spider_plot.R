library(ggplot2)
library(dplyr)
library(tidyr)

# Install ggradar if needed
if (!requireNamespace("ggradar", quietly = TRUE)) {
  if (!requireNamespace("devtools", quietly = TRUE)) {
    install.packages("devtools", repos = "https://cloud.r-project.org")
  }
  devtools::install_github("ricardo-bion/ggradar")
}
library(ggradar)

# Data for the 4 main models (from STATS_SUMMARY.txt and generate_figures.R)
# These values come from the full analysis
model_data <- data.frame(
  Model = c("GPT-5.2", "Grok 4.1", "Claude Sonnet 4", "Gemini 3 Pro"),
  Defensibility = c(0.81, 0.77, 0.73, 0.67),
  Reasoning = c(0.87, 0.87, 0.79, 0.76),
  Readability = c(17.9, 16.2, 17.5, 15.8),  # Lower is better (simpler)
  Length = c(3367, 2112, 1402, 1467),  # Lower is better (more concise)
  Cost = c(2526, 1518, 718, 368),  # Lower is better
  Procedures = c(7.0, 4.0, 2.0, 1.0)  # Lower is better (fewer procedures)
)

# Normalize all to 0-1 where 1 = best (outer edge = better)
radar_data <- model_data %>%
  mutate(
    # Defensibility: higher = better
    def_norm = (Defensibility - min(Defensibility)) / (max(Defensibility) - min(Defensibility)),
    # Reasoning: higher = better
    reasoning_norm = (Reasoning - min(Reasoning)) / (max(Reasoning) - min(Reasoning)),
    # Readability inverted: lower = simpler = better
    readability_norm = (max(Readability) - Readability) / (max(Readability) - min(Readability)),
    # Conciseness inverted: shorter = more concise = better
    conciseness_norm = (max(Length) - Length) / (max(Length) - min(Length)),
    # Cost inverted: lower cost = better = outer edge
    cost_norm = (max(Cost) - Cost) / (max(Cost) - min(Cost)),
    # Procedures inverted: fewer procedures = better = outer edge
    procedures_norm = (max(Procedures) - Procedures) / (max(Procedures) - min(Procedures))
  ) %>%
  select(Model, def_norm, reasoning_norm, readability_norm, conciseness_norm, cost_norm) %>%
  rename(
    group = Model,
    Defensibility = def_norm,
    Reasoning = reasoning_norm,
    Readability = readability_norm,
    Conciseness = conciseness_norm,
    `Cost Efficiency` = cost_norm
  )

# Colors for the 4 models (consistent with main figures)
model_colors <- c(
  "GPT-5.2" = "#1f77b4",
  "Grok 4.1" = "#ff7f0e",
  "Claude Sonnet 4" = "#2ca02c",
  "Gemini 3 Pro" = "#8c564b"
)

# Get colors in same order as radar data
radar_color_values <- model_colors[radar_data$group]

# Create the spider/radar plot
spider_plot <- ggradar(
  radar_data,
  grid.min = 0,
  grid.mid = 0.5,
  grid.max = 1,
  grid.label.size = 4,
  axis.label.size = 4,
  group.line.width = 1.2,
  group.point.size = 3,
  group.colours = radar_color_values,
  background.circle.colour = "white",
  gridline.mid.colour = "grey70",
  legend.position = "bottom",
  legend.text.size = 10
) +
  theme(
    legend.margin = margin(10, 0, 0, 0),
    plot.margin = margin(10, 10, 10, 10),
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold")
  ) +
  labs(title = "Multi-Dimensional Model Comparison")

# Save the plot
ggsave("output/spider_plot_4models.pdf", spider_plot, width = 10, height = 9)
ggsave("output/spider_plot_4models.png", spider_plot, width = 10, height = 9, dpi = 300)

print("Spider plot saved as spider_plot_4models.pdf and spider_plot_4models.png")
print(spider_plot)
