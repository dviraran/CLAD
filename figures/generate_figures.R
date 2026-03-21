# Generate figures for CLAD paper - VERSION 2 (Primary-Action Scoring)
# Run from figures/ directory
#
# This version uses primary-action scoring instead of majority vote:
# - Score 0: Primary action NOT met (regardless of secondary criteria)
# - Score 1: Primary action MET, <50% of secondary criteria met
# - Score 2: Primary action MET, ≥50% of secondary criteria met

library(tidyverse)
library(scales)
library(ggrepel)
library(patchwork)
library(treemapify)

# Install ggradar if needed
if (!requireNamespace("ggradar", quietly = TRUE)) {
  if (!requireNamespace("devtools", quietly = TRUE)) {
    install.packages("devtools", repos = "https://cloud.r-project.org")
  }
  devtools::install_github("ricardo-bion/ggradar")
}
library(ggradar)

# Set theme
theme_set(theme_minimal(base_size = 11) +
            theme(
              panel.grid.minor = element_blank(),
              plot.title = element_text(face = "bold", size = 12),
              plot.subtitle = element_text(color = "gray40", size = 10)
            ))

# Color palette - CONSISTENT across all figures
llm_colors <- c(
  "GPT-5.2" = "#1f77b4",
  "GPT-5.2-Concise" = "#aec7e8",
  "GPT-4o" = "#e377c2",
  "GPT-4o Mini" = "#f7b6d2",
  "Grok 4.1" = "#ff7f0e",
  "Claude Sonnet 4.5" = "#006400",
  "Claude Sonnet 4" = "#2ca02c",
  "Claude 3.5 Haiku" = "#98df8a",
  "Gemini 3 Pro" = "#8c564b",
  "Gemini 2.0 Flash" = "#c49c94",
  "Llama 4 Maverick" = "#d62728",
  "Llama 4 Scout" = "#ff9896",
  "Qwen3 30B" = "#c5b0d5",
  "Qwen 2.5 72B" = "#9467bd",
  "DeepSeek R1" = "#17becf",
  "Mistral Large" = "#bcbd22"
)

score_colors <- c("0" = "#e74c3c", "0.5" = "#f39c12", "1" = "#2ecc71")

# Load data
# runs.csv now includes:
# - primary_score: Score from primary-action scoring method (NEW in v2)
# - majority_score: majority vote of Claude, Grok, GPT-5.2 (used in v1)
# - mean_score: average of Claude, Grok, GPT-5.2
runs_raw <- read_csv("../results/runs.csv")

# Exclude non-clinical malpractice types
# - professional_boundaries_violation: ethics violations, not clinical decisions
# - documentation_failure: procedural, not clinical decision-making
# - equipment_or_facility_safety: administrative/facility issues
# - care_planning_error: policy/administrative issues
excluded_types <- c(
  "professional_boundaries_violation",
  "documentation_failure",
  "equipment_or_facility_safety",
  "care_planning_error"
)
runs_raw <- runs_raw %>%
  filter(!malpractice_type %in% excluded_types)

# Filter to main LLMs (>= 20 runs)
llm_counts <- runs_raw %>% count(llm_name)
main_llms <- llm_counts %>% filter(n >= 20) %>% pull(llm_name)

runs_all <- runs_raw %>%
  filter(llm_name %in% main_llms) %>%
  mutate(
    # VERSION 2: Use PRIMARY-ACTION scoring
    # Primary-action scoring: primary criterion must be met, then secondary threshold
    # Fall back to original score_0_2 if primary_score not available
    score = coalesce(primary_score, score_0_2) / 2,
    score_avg = coalesce(mean_score, score_0_2) / 2,  # For supplementary analysis (still uses original)
    llm_display = case_when(
      str_detect(llm_name, "gpt-5.2-concise") ~ "GPT-5.2-Concise",
      str_detect(llm_name, "gpt-5") ~ "GPT-5.2",
      str_detect(llm_name, "gpt-4o-mini") ~ "GPT-4o Mini",
      str_detect(llm_name, "gpt-4o") ~ "GPT-4o",
      str_detect(llm_name, "claude-sonnet-4\\.5") ~ "Claude Sonnet 4.5",
      str_detect(llm_name, "claude-sonnet-4") ~ "Claude Sonnet 4",
      str_detect(llm_name, "claude-3.5-haiku") ~ "Claude 3.5 Haiku",
      str_detect(llm_name, "gemini-3-pro") ~ "Gemini 3 Pro",
      str_detect(llm_name, "gemini-2.0-flash") ~ "Gemini 2.0 Flash",
      str_detect(llm_name, "llama-4-maverick") ~ "Llama 4 Maverick",
      str_detect(llm_name, "llama-4-scout") ~ "Llama 4 Scout",
      str_detect(llm_name, "qwen3") ~ "Qwen3 30B",
      str_detect(llm_name, "qwen-2.5") ~ "Qwen 2.5 72B",
      str_detect(llm_name, "grok") ~ "Grok 4.1",
      str_detect(llm_name, "deepseek") ~ "DeepSeek R1",
      str_detect(llm_name, "mistral") ~ "Mistral Large",
      TRUE ~ llm_name
    ),
    malpractice_display = str_replace_all(malpractice_type, "_", " ") %>%
      str_to_title()
  )

# Use only FIRST run per case-LLM pair for main analyses
# (duplicate runs are used only for test-retest analysis)
runs_first <- runs_all %>%
  arrange(started_at) %>%
  group_by(case_id, llm_display) %>%
  slice(1) %>%
  ungroup()

# Count NA runs (score_valid=FALSE) before filtering
n_na_runs <- sum(!runs_first$score_valid, na.rm = TRUE)
n_total_runs <- nrow(runs_first)

# Identify cases where NO model scored >0 (treating NA as 0)
# This includes: all-NA cases, all-zero cases, AND cases with <3 valid evals that all scored 0
case_score_summary <- runs_first %>%
  mutate(score_for_check = ifelse(score_valid, score, 0)) %>%
  group_by(case_id) %>%
  summarise(
    n_total_runs = n(),
    n_valid_runs = sum(score_valid, na.rm = TRUE),
    n_above_zero = sum(score_for_check > 0, na.rm = TRUE),
    max_score = max(score_for_check, na.rm = TRUE),
    .groups = "drop"
  )

# Cases to exclude: any case where no model scored >0
excluded_cases <- case_score_summary %>%
  filter(n_above_zero == 0) %>%
  pull(case_id)

cat(sprintf("Excluding %d cases where no model scored >0 (all zeros or NA)\n",
            length(excluded_cases)))

# Filter to valid scores only (exclude NA/deferred cases and bad cases)
runs <- runs_first %>%
  filter(score_valid == TRUE) %>%
  filter(!case_id %in% excluded_cases)

# For backward compatibility, also define these (used in difficulty figure)
all_na_cases <- case_score_summary %>% filter(n_valid_runs == 0) %>% pull(case_id)
all_zero_cases <- excluded_cases  # Now includes all 36 cases

# Count duplicate pairs for reporting
n_duplicate_pairs <- runs_all %>%
  count(case_id, llm_display) %>%
  filter(n > 1) %>%
  nrow()

# Count cases and runs
n_cases <- n_distinct(runs$case_id)
n_runs <- nrow(runs)
n_llms <- n_distinct(runs$llm_display)

cat(sprintf("\n=== Dataset Summary ===\n"))
cat(sprintf("Cases: %d\n", n_cases))
cat(sprintf("Total case-LLM runs (first run only): %d\n", n_total_runs))
cat(sprintf("NA runs (did not reach decision): %d\n", n_na_runs))
cat(sprintf("Valid runs for analysis: %d\n", n_runs))
cat(sprintf("Total runs (including duplicates): %d\n", nrow(runs_all)))
cat(sprintf("Duplicate pairs (for test-retest): %d\n", n_duplicate_pairs))
cat(sprintf("LLMs: %d\n", n_llms))

# ============================================
# Load and merge cost data
# ============================================

cost_raw <- read_csv("../results/cost_summary.csv", show_col_types = FALSE)

# Create llm_display mapping for cost data
cost_data <- cost_raw %>%
  mutate(
    llm_display = case_when(
      str_detect(llm_name, "gpt-5.2-concise") ~ "GPT-5.2-Concise",
      str_detect(llm_name, "gpt-5") ~ "GPT-5.2",
      str_detect(llm_name, "gpt-4o-mini") ~ "GPT-4o Mini",
      str_detect(llm_name, "gpt-4o") ~ "GPT-4o",
      str_detect(llm_name, "claude-sonnet-4\\.5") ~ "Claude Sonnet 4.5",
      str_detect(llm_name, "claude-sonnet-4") ~ "Claude Sonnet 4",
      str_detect(llm_name, "claude-3.5-haiku") ~ "Claude 3.5 Haiku",
      str_detect(llm_name, "gemini-3-pro") ~ "Gemini 3 Pro",
      str_detect(llm_name, "gemini-2.0-flash") ~ "Gemini 2.0 Flash",
      str_detect(llm_name, "llama-4-maverick") ~ "Llama 4 Maverick",
      str_detect(llm_name, "llama-4-scout") ~ "Llama 4 Scout",
      str_detect(llm_name, "qwen3") ~ "Qwen3 30B",
      str_detect(llm_name, "qwen-2.5") ~ "Qwen 2.5 72B",
      str_detect(llm_name, "grok") ~ "Grok 4.1",
      str_detect(llm_name, "deepseek") ~ "DeepSeek R1",
      str_detect(llm_name, "mistral") ~ "Mistral Large",
      TRUE ~ llm_name
    ),
    # Fill NA costs with 0 for runs with no procedures
    total_cost = coalesce(total_cost_negotiated, 0)
  )

# Merge cost data with runs
runs <- runs %>%
  left_join(
    cost_data %>% select(run_id, procedure_count, matched_count, match_rate, total_cost),
    by = "run_id"
  ) %>%
  mutate(
    procedure_count = coalesce(procedure_count, 0),
    total_cost = coalesce(total_cost, 0)
  )

cat(sprintf("\n=== Cost Data Summary ===\n"))
cat(sprintf("Runs with cost data: %d\n", sum(!is.na(runs$total_cost))))
cat(sprintf("Mean cost per recommendation: $%.2f\n", mean(runs$total_cost, na.rm = TRUE)))
cat(sprintf("Mean procedure count: %.1f\n", mean(runs$procedure_count, na.rm = TRUE)))

# Aggregate LLM summary
llm_summary <- runs %>%
  group_by(llm_display) %>%
  summarise(
    n = n(),
    mean_score = mean(score, na.rm = TRUE),
    defensible_rate = mean(score == 1, na.rm = TRUE),
    partial_rate = mean(score == 0.5, na.rm = TRUE),
    liability_rate = mean(score == 0, na.rm = TRUE),
    mean_length = mean(recommendation_length, na.rm = TRUE),
    mean_reasoning = mean(reasoning_quality_score, na.rm = TRUE),
    mean_met_criteria = mean(met_criteria_count, na.rm = TRUE),
    mean_readability = mean(transformer_readability_score, na.rm = TRUE),
    mean_fk = mean(flesch_kincaid_grade, na.rm = TRUE),
    mean_cost = mean(total_cost, na.rm = TRUE),
    mean_procedures = mean(procedure_count, na.rm = TRUE),
    n_defensible = sum(score == 1, na.rm = TRUE),
    total_cost_defensible = sum(total_cost[score == 1], na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    # Cost per Defensible Recommendation (CPDR)
    cpdr = ifelse(n_defensible > 0, total_cost_defensible / n_defensible, NA)
  ) %>%
  arrange(desc(mean_score))

# ============================================
# FIGURE 1: Combined Overview Figure
# ============================================

# Panel 1A: Simple treemap by malpractice type only
# IMPORTANT: Use hardcoded canonical counts to match Supplementary Table 2
# (runs data has inconsistent malpractice types across different runs of same case)
treemap_simple <- tribble(
  ~malpractice_display, ~n,
  "Diagnosis Delay Or Error", 75,
  "Test Selection Error", 48,
  "Referral Failure", 23,
  "Surgical Technique Error", 18,
  "Informed Consent", 17,
  "Treatment Timing Error", 14,
  "Medication Selection Error", 11,
  "Other", 10,
  "Discharge Disposition Error", 9,
  "Monitoring Or Escalation Failure", 5,
  "Care Management Error", 3
) %>%
  mutate(
    malpractice_short = case_when(
      malpractice_display == "Diagnosis Delay Or Error" ~ "Diagnosis\nError",
      malpractice_display == "Test Selection Error" ~ "Test\nSelection",
      malpractice_display == "Treatment Timing Error" ~ "Treatment\nTiming",
      malpractice_display == "Monitoring Or Escalation Failure" ~ "Monitoring\nFailure",
      malpractice_display == "Surgical Technique Error" ~ "Surgical\nError",
      malpractice_display == "Discharge Disposition Error" ~ "Discharge\nError",
      malpractice_display == "Medication Selection Error" ~ "Medication\nError",
      malpractice_display == "Care Management Error" ~ "Care\nManagement",
      malpractice_display == "Referral Failure" ~ "Referral\nFailure",
      malpractice_display == "Informed Consent" ~ "Informed\nConsent",
      malpractice_display == "Professional Boundaries Violation" ~ "Professional\nBoundaries",
      TRUE ~ str_wrap(malpractice_display, 10)
    )
  )

fig1a_treemap <- ggplot(treemap_simple, aes(area = n, fill = n, label = paste0(malpractice_short, "\n(", n, ")"))) +
  geom_treemap(color = "white", size = 2) +
  geom_treemap_text(color = "white", place = "centre", size = 8, fontface = "bold", grow = FALSE) +
  scale_fill_gradient(low = "#74b9ff", high = "#0984e3", guide = "none") +
  theme_void()

# Panel 1B: Load external image (workflow diagram)
library(png)
library(grid)

# Read the high-res PNG image and wrap it for patchwork
fig1b_img <- png::readPNG("figure1b_hires.png")
fig1b_grob <- grid::rasterGrob(fig1b_img, interpolate = TRUE)

# Use wrap_elements for patchwork compatibility
fig1b_consultation <- wrap_elements(full = fig1b_grob)

# Panel 1C: Criteria heatmap + 3 judge scores + majority vote
# PRIMARY-ACTION SCORING: Urgent Referral is the PRIMARY criterion (marked with *)
# If primary criterion NOT met → Score 0 (regardless of other criteria)
# If primary criterion MET → Score based on secondary criteria (0.5 if <50%, 1 if >=50%)
example_models <- data.frame(
  model = c("GPT-5.2", "Grok 4.1", "Claude\nSonnet 4", "DeepSeek\nR1", "Mistral\nLarge", "Gemini\n3 Pro", "Claude 3.5\nHaiku", "Qwen\n2.5 72B", "Gemini 2.0\nFlash", "Llama 4\nScout", "Llama 4\nMaverick", "GPT-4o", "GPT-4o\nMini"),
  # PRIMARY-ACTION SCORES: primary must be met for any positive score
  score = c(1, 1, 1, 1, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0),
  # Criteria met by each model - Urgent Referral is PRIMARY
  urgent_referral = c(TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE),  # PRIMARY
  reasoning = c(TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE),  # secondary
  followup = c(TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE),  # secondary
  # Judge scores under primary-action: miss primary = 0
  claude_score = c(1, 1, 1, 1, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0),
  grok_score = c(1, 1, 1, 1, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0),
  gpt5_score = c(1, 1, 1, 1, 1, 0.5, 0.5, 0, 0, 0, 0, 0, 0)
) %>%
  mutate(
    score_label = case_when(score == 1 ~ "1", score == 0.5 ~ "0.5", TRUE ~ "0"),
    model = factor(model, levels = rev(model))  # Reverse for top-to-bottom order
  )

# Create long format for criteria heatmap
# Mark Urgent Referral as PRIMARY criterion with asterisk
criteria_long <- example_models %>%
  select(model, urgent_referral, reasoning, followup) %>%
  pivot_longer(-model, names_to = "criterion", values_to = "met") %>%
  mutate(
    criterion = case_when(
      criterion == "urgent_referral" ~ "Urgent\nReferral*",  # * marks PRIMARY
      criterion == "reasoning" ~ "Clinical\nReasoning",
      criterion == "followup" ~ "Follow-up\nPlan"
    ),
    criterion = factor(criterion, levels = c("Urgent\nReferral*", "Clinical\nReasoning", "Follow-up\nPlan"))
  )

# Create long format for judge scores
judges_long <- example_models %>%
  select(model, claude_score, grok_score, gpt5_score) %>%
  pivot_longer(-model, names_to = "judge", values_to = "judge_score") %>%
  mutate(
    judge = case_when(
      judge == "claude_score" ~ "Claude",
      judge == "grok_score" ~ "Grok",
      judge == "gpt5_score" ~ "GPT-5.2"
    ),
    judge = factor(judge, levels = c("Claude", "Grok", "GPT-5.2")),
    score_label = case_when(judge_score == 1 ~ "1", judge_score == 0.5 ~ "0.5", TRUE ~ "0"),
    score_color = case_when(
      judge_score == 1 ~ "Defensible",
      judge_score == 0.5 ~ "Partial",
      TRUE ~ "Liability"
    )
  )

# Score data for the majority vote column
score_data <- example_models %>%
  select(model, score, score_label) %>%
  mutate(
    score_color = case_when(
      score == 1 ~ "Defensible",
      score == 0.5 ~ "Partial",
      TRUE ~ "Liability"
    )
  )

# Criteria heatmap (left side) - criteria met by each model's response
fig1c_criteria <- ggplot(criteria_long, aes(x = criterion, y = model, fill = met)) +
  geom_tile(color = "white", linewidth = 1.2) +
  geom_text(aes(label = ifelse(met, "YES", "NO")),
            color = "white", size = 3.5, fontface = "bold") +
  scale_fill_manual(values = c("TRUE" = "#27ae60", "FALSE" = "#e74c3c"),
                    labels = c("TRUE" = "Met", "FALSE" = "Not Met"),
                    name = NULL) +
  labs(x = NULL, y = NULL, title = "Criteria (* = primary)") +
  theme_minimal() +
  theme(
    panel.grid = element_blank(),
    legend.position = "none",
    axis.text.x = element_text(size = 8),
    axis.text.y = element_text(size = 9),
    plot.title = element_text(size = 10, face = "bold", hjust = 0.5)
  )

# Judge scores heatmap (middle) - shows 0, 0.5, 1
colors_score <- c("Defensible" = "#27ae60", "Partial" = "#f39c12", "Liability" = "#e74c3c")

fig1c_judges <- ggplot(judges_long, aes(x = judge, y = model, fill = score_color)) +
  geom_tile(color = "white", linewidth = 1.2) +
  geom_text(aes(label = score_label),
            color = "white", size = 3.5, fontface = "bold") +
  scale_fill_manual(values = colors_score, guide = "none") +
  labs(x = NULL, y = NULL, title = "LLM Judges") +
  theme_minimal() +
  theme(
    panel.grid = element_blank(),
    legend.position = "none",
    axis.text.x = element_text(size = 8),
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    plot.title = element_text(size = 10, face = "bold", hjust = 0.5)
  )

# Majority vote score column (right side)
fig1c_score <- ggplot(score_data, aes(x = "Majority\nVote", y = model, fill = score_color)) +
  geom_tile(color = "white", linewidth = 1.2) +
  geom_text(aes(label = score_label), color = "white", size = 3.5, fontface = "bold") +
  scale_fill_manual(values = colors_score, guide = "none") +
  scale_x_discrete(position = "bottom") +
  labs(x = NULL, y = NULL, title = "Score") +
  theme_minimal() +
  theme(
    panel.grid = element_blank(),
    axis.text.x = element_text(size = 8),
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    plot.title = element_text(size = 10, face = "bold", hjust = 0.5)
  )

# Combine: criteria (3 cols) + judges (3 cols) + majority (1 col)
fig1c_combined <- wrap_plots(fig1c_criteria, fig1c_judges, fig1c_score, widths = c(3, 3, 1))

# Combine Figure 1: A (top-left), B (right spanning 2 rows), C (bottom-left)
# Use design layout
design <- "
AB
CB
"

fig1_combined <- fig1a_treemap + fig1b_consultation + wrap_elements(full = fig1c_combined) +
  plot_layout(design = design, widths = c(1, 1.2), heights = c(1, 1)) +
  plot_annotation(tag_levels = "A")

ggsave("output/fig1_combined.pdf", fig1_combined, width = 12, height = 9)

# ============================================
# FIGURE 2: Defensibility Results
# ============================================

# Panel 2A: Mean defensibility score by model
fig2a_defensibility <- llm_summary %>%
  mutate(llm_display = fct_reorder(llm_display, mean_score)) %>%
  ggplot(aes(x = llm_display, y = mean_score)) +
  geom_col(aes(fill = mean_score), alpha = 0.9, width = 0.7) +
  geom_text(aes(label = sprintf("%.2f", mean_score)),
            hjust = -0.1, size = 3.5, fontface = "bold") +
  coord_flip() +
  scale_fill_gradient(low = "#e74c3c", high = "#2ecc71", guide = "none") +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    x = NULL,
    y = "Mean Defensibility Score (0-1)"
  )

# Panel 2B: Performance by malpractice category
malpractice_perf <- runs %>%
  group_by(malpractice_display) %>%
  summarise(
    n_cases = n_distinct(case_id),
    n_runs = n(),
    mean_score = mean(score, na.rm = TRUE),
    defensible_rate = mean(score == 1, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  filter(n_runs >= 10) %>%  # Only show categories with enough data
  mutate(malpractice_display = fct_reorder(malpractice_display, mean_score))

# Get best and worst for subtitle
best_mal <- malpractice_perf %>% slice_max(mean_score, n = 1)
worst_mal <- malpractice_perf %>% slice_min(mean_score, n = 1)

fig2b_malpractice <- ggplot(malpractice_perf, aes(x = malpractice_display, y = mean_score)) +
  geom_col(aes(fill = mean_score), alpha = 0.9, width = 0.7) +
  geom_text(aes(label = sprintf("%.2f", mean_score)), hjust = -0.1, size = 3) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "gray50") +
  coord_flip() +
  scale_fill_gradient(low = "#e74c3c", high = "#2ecc71", guide = "none") +
  labs(
    x = NULL,
    y = "Mean Defensibility Score"
  ) +
  expand_limits(y = 1)

# Panel 2C: Defensibility vs Reasoning Quality (MODEL-LEVEL)
reasoning_summary <- llm_summary %>%
  filter(!is.na(mean_reasoning))

# Calculate model-level correlation for annotation
reasoning_cor_model <- cor(reasoning_summary$mean_reasoning, reasoning_summary$mean_score, use = "complete.obs", method = "spearman")

fig2c_reasoning <- ggplot(reasoning_summary, aes(x = mean_reasoning, y = mean_score)) +
  geom_point(aes(color = llm_display), size = 5) +
  geom_smooth(method = "lm", se = TRUE, color = "gray50", linetype = "dashed", alpha = 0.3) +
  geom_text(aes(label = llm_display), vjust = -1.2, size = 3) +
  scale_color_manual(values = llm_colors, guide = "none") +
  scale_x_continuous(limits = c(0.6, 0.9)) +
  scale_y_continuous(limits = c(0.4, 0.9)) +
  labs(
    x = "Mean Reasoning Quality Score",
    y = "Mean Defensibility Score (0-1)"
  ) +
  annotate("text", x = 0.63, y = 0.88,
           label = sprintf("r = %.2f", reasoning_cor_model),
           size = 4, fontface = "bold")

# Combine Figure 2: All three panels in one row
fig2_combined <- fig2a_defensibility | fig2b_malpractice | fig2c_reasoning +
  plot_annotation(tag_levels = "A")

ggsave("output/fig2_defensibility.pdf", fig2_combined, width = 16, height = 5)

# ============================================
# FIGURE 3: Readability and Trade-off
# ============================================

# Panel 3A: Response length by model (boxplots to show variation)
# Order by median length for consistency
length_order <- runs %>%
  group_by(llm_display) %>%
  summarise(median_length = median(recommendation_length, na.rm = TRUE)) %>%
  arrange(median_length) %>%
  pull(llm_display)

fig3a_length <- runs %>%
  mutate(llm_display = factor(llm_display, levels = length_order)) %>%
  ggplot(aes(x = llm_display, y = recommendation_length, fill = llm_display)) +
  geom_boxplot(alpha = 0.8, outlier.size = 0.8, outlier.alpha = 0.5) +
  coord_flip() +
  scale_fill_manual(values = llm_colors, guide = "none") +
  scale_y_continuous(labels = comma_format()) +
  labs(
    x = NULL,
    y = "Characters"
  ) +
  theme(legend.position = "none")

# Panel 3B: Readability (Transformer Score - DeBERTa)
fig3b_readability <- llm_summary %>%
  filter(!is.na(mean_readability)) %>%
  mutate(llm_display_ordered = fct_reorder(llm_display, mean_readability)) %>%
  ggplot(aes(x = llm_display_ordered, y = mean_readability)) +
  geom_col(aes(fill = defensible_rate), alpha = 0.9, width = 0.7) +
  geom_text(aes(label = sprintf("%.1f", mean_readability)), hjust = -0.1, size = 3) +
  coord_flip() +
  scale_fill_gradient(low = "#e74c3c", high = "#2ecc71", name = "Defensibility") +
  labs(
    x = NULL,
    y = "Readability Score\n(higher = more complex)"
  ) +
  expand_limits(y = 20) +
  theme(legend.position = "none")

# Panel 3C: Readability vs Defensibility scatter plot
# Use CONSISTENT colors with the radar chart
fig3c_scatter <- llm_summary %>%
  filter(!is.na(mean_readability)) %>%
  ggplot(aes(x = mean_readability, y = mean_score)) +
  geom_point(aes(color = llm_display), size = 4, alpha = 0.8) +
  geom_smooth(method = "lm", se = TRUE, color = "gray50", linetype = "dashed", alpha = 0.3) +
  geom_text_repel(aes(label = llm_display), size = 2.8, box.padding = 0.3,
                  point.padding = 0.2, max.overlaps = 10, seed = 42,
                  min.segment.length = 0) +
  scale_color_manual(values = llm_colors, guide = "none") +
  labs(
    x = "Readability Score (higher = more complex)",
    y = "Defensibility (0-1)"
  ) +
  coord_cartesian(xlim = c(15, 18), ylim = c(0.4, 0.9)) +
  theme(plot.margin = margin(5, 10, 5, 5))

# Panel 3D: Readability metrics correlation with defensibility
# Show which readability dimensions predict clinical success (matching RMD 7.5.5)
readability_cors <- data.frame(
  metric = c("Transformer Readability", "Semantic Coherence (Local)", "Flesch-Kincaid Grade",
             "Semantic Coherence (Global)", "SMOG Index", "Lexical Overlap (Global)",
             "Lexical Overlap (Adjacent)", "Pronoun Density"),
  correlation = c(
    cor(runs$transformer_readability_score, runs$score, use = "complete.obs"),
    cor(runs$semantic_coherence_local, runs$score, use = "complete.obs"),
    cor(runs$flesch_kincaid_grade, runs$score, use = "complete.obs"),
    cor(runs$semantic_coherence_global, runs$score, use = "complete.obs"),
    cor(runs$smog_index, runs$score, use = "complete.obs"),
    cor(runs$lexical_overlap_global, runs$score, use = "complete.obs"),
    cor(runs$lexical_overlap_adjacent, runs$score, use = "complete.obs"),
    cor(runs$pronoun_density, runs$score, use = "complete.obs")
  )
) %>%
  mutate(
    metric = fct_reorder(metric, correlation),
    direction = ifelse(correlation > 0, "Positive", "Negative")
  )

fig3d_readability_cors <- ggplot(readability_cors, aes(x = metric, y = correlation, fill = direction)) +
  geom_col(alpha = 0.9, width = 0.7) +
  geom_text(aes(label = sprintf("%.3f", correlation)),
            hjust = ifelse(readability_cors$correlation > 0, -0.1, 1.1), size = 3) +
  coord_flip() +
  scale_fill_manual(values = c("Positive" = "#2ecc71", "Negative" = "#e74c3c"), guide = "none") +
  geom_hline(yintercept = 0, linetype = "solid", color = "gray40") +
  labs(
    x = NULL,
    y = "Pearson Correlation with Defensibility Score"
  ) +
  scale_y_continuous(limits = c(-0.35, 0.25))

# Combine Figure 3 with better layout - 3D before 3C in bottom row
fig3_combined <- (fig3a_length | fig3b_readability) /
  (fig3d_readability_cors | fig3c_scatter) +
  plot_layout(heights = c(0.9, 1.1)) +
  plot_annotation(tag_levels = "A")

ggsave("output/fig3_readability.pdf", fig3_combined, width = 13, height = 10)

# ============================================
# FIGURE 4: Cost Analysis and Multi-dimensional Trade-offs
# ============================================

# Panel 4A: Cost by model (violin/box plot)
# Count outliers above 20K for caption
n_outliers_20k <- runs %>% filter(total_cost > 20000) %>% nrow()
outlier_models <- runs %>% filter(total_cost > 20000) %>% count(llm_display) %>%
  mutate(label = paste0(llm_display, " (n=", n, ")")) %>% pull(label) %>% paste(collapse = ", ")

fig4a_cost <- runs %>%
  mutate(llm_display = factor(llm_display, levels = llm_summary %>%
                                arrange(mean_cost) %>% pull(llm_display))) %>%
  ggplot(aes(x = llm_display, y = total_cost, fill = llm_display)) +
  geom_violin(alpha = 0.7, scale = "width") +
  geom_boxplot(width = 0.15, fill = "white", alpha = 0.8, outlier.size = 0.5) +
  coord_flip(ylim = c(0, 10000)) +
  scale_fill_manual(values = llm_colors, guide = "none") +
  scale_y_continuous(labels = scales::dollar_format()) +
  labs(
    x = NULL,
    y = "Total Cost per Recommendation ($)",
    caption = sprintf("X-axis truncated at $10K; %d outliers omitted (%s)", n_outliers_20k, outlier_models)
  ) +
  theme(plot.caption = element_text(size = 8, color = "gray50"))

# Panel 4B: Cost vs Defensibility with Pareto Frontier
# Calculate 95% CI for error bars
cost_def_summary <- llm_summary %>%
  select(llm_display, mean_score, mean_cost, n) %>%
  mutate(
    se_score = sqrt(mean_score * (1 - mean_score) / n),
    se_cost = NA  # Would need to calculate from raw data
  )

# Identify Pareto-optimal models (maximize defensibility, minimize cost)
pareto_models <- llm_summary %>%
  arrange(mean_cost) %>%
  mutate(
    is_pareto = {
      max_def_so_far <- cummax(mean_score)
      mean_score >= max_def_so_far
    }
  ) %>%
  filter(is_pareto)

fig4b_pareto <- ggplot(llm_summary, aes(x = mean_cost, y = mean_score)) +
  geom_point(aes(color = llm_display), size = 5, alpha = 0.8) +
  geom_text_repel(aes(label = llm_display), size = 3, box.padding = 0.3,
                  point.padding = 0.2, max.overlaps = 10, seed = 42) +
  # Add Pareto frontier line
  geom_line(data = pareto_models %>% arrange(mean_cost),
            aes(x = mean_cost, y = mean_score),
            linetype = "dashed", color = "gray40", linewidth = 0.8) +
  scale_color_manual(values = llm_colors, guide = "none") +
  scale_x_continuous(labels = scales::dollar_format()) +
  labs(
    x = "Mean Cost per Recommendation ($)",
    y = "Mean Defensibility Score (0-1)"
  ) +
  annotate("text", x = max(llm_summary$mean_cost) * 0.7, y = 0.4,
           label = "Pareto Frontier", fontface = "italic", size = 3, color = "gray40")

# Panel 4C: Spider plot with cost and procedures (moved from Fig 3)
# Filter to models with complete data for radar plot
radar_data <- llm_summary %>%
  filter(!is.na(mean_readability) & !is.na(mean_reasoning) & !is.na(mean_cost) & !is.na(mean_procedures)) %>%
  mutate(
    # Normalize all to 0-1 where 1 = best (outer edge = better)
    score_norm = (mean_score - min(mean_score)) / (max(mean_score) - min(mean_score)),
    reasoning_norm = (mean_reasoning - min(mean_reasoning, na.rm=TRUE)) / (max(mean_reasoning, na.rm=TRUE) - min(mean_reasoning, na.rm=TRUE)),
    # Readability inverted: lower = simpler = better
    readability_norm = (max(mean_readability, na.rm=TRUE) - mean_readability) / (max(mean_readability, na.rm=TRUE) - min(mean_readability, na.rm=TRUE)),
    # Conciseness inverted: shorter = more concise = better
    conciseness_norm = (max(mean_length) - mean_length) / (max(mean_length) - min(mean_length)),
    # Cost inverted: lower cost = better = outer edge
    cost_norm = (max(mean_cost) - mean_cost) / (max(mean_cost) - min(mean_cost)),
    # Procedures inverted: fewer procedures = better = outer edge
    procedures_norm = (max(mean_procedures) - mean_procedures) / (max(mean_procedures) - min(mean_procedures))
  ) %>%
  select(llm_display, score_norm, reasoning_norm, readability_norm, conciseness_norm, cost_norm, procedures_norm) %>%
  rename(
    group = llm_display,
    Defensibility = score_norm,
    Reasoning = reasoning_norm,
    Readability = readability_norm,
    Conciseness = conciseness_norm,
    `Cost Efficiency` = cost_norm,
    `Procedure Efficiency` = procedures_norm
  )

# Split into two groups: proprietary frontier vs open-source & mini
proprietary_frontier <- c("GPT-5.2", "GPT-5.2-Concise", "GPT-4o", "Claude Sonnet 4.5", "Claude Sonnet 4", "Gemini 3 Pro", "Grok 4.1")
open_mini <- c("DeepSeek R1", "Mistral Large", "Llama 4 Maverick", "Llama 4 Scout",
               "Qwen3 30B", "Qwen 2.5 72B", "GPT-4o Mini", "Claude 3.5 Haiku", "Gemini 2.0 Flash")

radar_data_a <- radar_data %>% filter(group %in% proprietary_frontier)
radar_data_b <- radar_data %>% filter(group %in% open_mini)

# Panel A: Proprietary frontier
radar_colors_a <- llm_colors[radar_data_a$group]
fig5_radar_a <- ggradar(
  radar_data_a,
  grid.min = 0, grid.mid = 0.5, grid.max = 1,
  grid.label.size = 3, axis.label.size = 3,
  group.line.width = 0.8, group.point.size = 2,
  group.colours = radar_colors_a,
  background.circle.colour = "white",
  gridline.mid.colour = "grey70",
  legend.position = "right",
  legend.text.size = 7
) +
  theme(legend.margin = margin(0, 0, 0, 0), plot.margin = margin(5, 5, 5, 5))

# Panel B: Open-source & mini
radar_colors_b <- llm_colors[radar_data_b$group]
fig5_radar_b <- ggradar(
  radar_data_b,
  grid.min = 0, grid.mid = 0.5, grid.max = 1,
  grid.label.size = 3, axis.label.size = 3,
  group.line.width = 0.8, group.point.size = 2,
  group.colours = radar_colors_b,
  background.circle.colour = "white",
  gridline.mid.colour = "grey70",
  legend.position = "right",
  legend.text.size = 7
) +
  theme(legend.margin = margin(0, 0, 0, 0), plot.margin = margin(5, 5, 5, 5))

# Combined two-panel spider (for standalone Fig 5)
fig5_combined <- fig5_radar_a + fig5_radar_b +
  plot_annotation(tag_levels = "A") +
  plot_layout(ncol = 2)

# Also keep single radar for fig4 combined (use panel A as representative)
fig4c_radar <- fig5_radar_a

# Panel 4D: Cost per Defensible Recommendation (CPDR) bar chart
fig4d_cpdr <- llm_summary %>%
  filter(!is.na(cpdr) & cpdr > 0) %>%
  mutate(llm_display = fct_reorder(llm_display, -cpdr)) %>%
  ggplot(aes(x = llm_display, y = cpdr, fill = mean_score)) +
  geom_col(alpha = 0.9, width = 0.7) +
  geom_text(aes(label = sprintf("$%.0f", cpdr)), hjust = -0.1, size = 3) +
  coord_flip() +
  scale_fill_gradient(low = "#e74c3c", high = "#2ecc71", name = "Score") +
  scale_y_continuous(labels = scales::dollar_format(), limits = c(0, 3500)) +
  labs(
    x = NULL,
    y = "Cost per Defensible Recommendation ($)"
  ) +
  theme(legend.position = "none")

# Combine Figure 4 - 3-panel (Pareto is now standalone Fig 3)
fig4_combined <- (fig4a_cost) /
  (fig4d_cpdr | fig4c_radar) +
  plot_layout(heights = c(1, 1.2)) +
  plot_annotation(tag_levels = "A")

ggsave("output/fig4_cost.pdf", fig4_combined, width = 15, height = 11)

# Standalone spider plot (Fig 5 in paper) - two panels
ggsave("output/fig5_spider.pdf", fig5_combined, width = 16, height = 6)
ggsave("output/fig5_spider.png", fig5_combined, width = 16, height = 6, dpi = 300)

# Supplementary: cost violin + CPDR (2-panel)
fig_supp_cost_cpdr <- (fig4a_cost) / (fig4d_cpdr) +
  plot_layout(heights = c(1, 1)) +
  plot_annotation(tag_levels = "A")
ggsave("output/fig_supp_cost_cpdr.pdf", fig_supp_cost_cpdr, width = 12, height = 10)
ggsave("output/fig_supp_cost_cpdr.png", fig_supp_cost_cpdr, width = 12, height = 10, dpi = 300)

# ============================================
# SUPPLEMENTARY TABLE 1: Cases by Jurisdiction x Malpractice
# ============================================

supp_table1 <- runs %>%
  distinct(case_id, .keep_all = TRUE) %>%
  count(malpractice_display, jurisdiction) %>%
  pivot_wider(names_from = jurisdiction, values_from = n, values_fill = 0) %>%
  # Only include jurisdictions that exist in data
  mutate(Total = rowSums(across(where(is.numeric)))) %>%
  arrange(desc(Total)) %>%
  rename(`Malpractice Type` = malpractice_display)

# Save as CSV
write_csv(supp_table1, "figures/supp_table1_cases.csv")

# Print for LaTeX
cat("\n=== Supplementary Table 1: Cases by Jurisdiction ===\n")
print(supp_table1)

# ============================================
# SUPPLEMENTARY FIGURE: Heatmap
# ============================================

heatmap_data <- runs %>%
  group_by(llm_display, malpractice_display) %>%
  summarise(mean_score = mean(score, na.rm = TRUE), n = n(), .groups = "drop") %>%
  filter(n >= 2)

# Get case counts for each malpractice type
mal_case_counts <- runs %>%
  group_by(malpractice_display) %>%
  summarise(n_cases = n_distinct(case_id), .groups = "drop")

# Order by mean score
llm_order_heatmap <- llm_summary %>% arrange(desc(mean_score)) %>% pull(llm_display)
mal_order_heatmap <- runs %>%
  group_by(malpractice_display) %>%
  summarise(mean_score = mean(score, na.rm = TRUE), .groups = "drop") %>%
  arrange(desc(mean_score)) %>%
  pull(malpractice_display)

# Create x-axis labels with case counts
mal_labels_with_counts <- mal_case_counts %>%
  mutate(label = paste0(malpractice_display, "\n(n=", n_cases, ")")) %>%
  select(malpractice_display, label)

# Check for NA cases by model and malpractice type (using runs_first before filtering)
na_summary <- runs_first %>%
  filter(!score_valid) %>%
  group_by(llm_display, malpractice_display) %>%
  summarise(n_na = n(), .groups = "drop")

# Note about care management
care_mgmt_na <- na_summary %>%
  filter(malpractice_display == "Care Management Error") %>%
  mutate(note = paste0(llm_display, " (", n_na, " NA)")) %>%
  pull(note) %>%
  paste(collapse = ", ")

fig_supp_heatmap <- ggplot(heatmap_data,
                           aes(x = factor(malpractice_display, levels = mal_order_heatmap),
                               y = factor(llm_display, levels = rev(llm_order_heatmap)),
                               fill = mean_score)) +
  geom_tile(color = "white", linewidth = 0.8) +
  geom_text(aes(label = sprintf("%.2f", mean_score)),
            color = ifelse(heatmap_data$mean_score > 0.6, "white", "black"), size = 3) +
  scale_fill_gradientn(
    colors = c("#e74c3c", "#f39c12", "#f1c40f", "#2ecc71"),
    values = scales::rescale(c(0, 0.4, 0.6, 1)),
    limits = c(0, 1),
    name = "Score"
  ) +
  scale_x_discrete(labels = function(x) {
    sapply(x, function(m) {
      n <- mal_case_counts$n_cases[mal_case_counts$malpractice_display == m]
      paste0(m, "\n(n=", n, ")")
    })
  }) +
  labs(
    title = "Performance Heatmap: LLM × Malpractice Type",
    subtitle = "Mean defensibility score (0-1 scale). Note: For Care Management Error, some models had only NA responses.",
    x = NULL,
    y = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 9),
    axis.text.y = element_text(size = 10),
    legend.position = "right",
    panel.grid = element_blank(),
    plot.title = element_text(face = "bold", size = 12),
    plot.subtitle = element_text(color = "gray40", size = 10)
  )

ggsave("output/fig_supp_heatmap.pdf", fig_supp_heatmap, width = 14, height = 7)

# ============================================
# SUPPLEMENTARY FIGURE 2: Performance by Jurisdiction
# ============================================

jurisdiction_perf <- runs %>%
  filter(!is.na(jurisdiction)) %>%
  group_by(jurisdiction) %>%
  summarise(
    n_cases = n_distinct(case_id),
    n_runs = n(),
    mean_score = mean(score, na.rm = TRUE),
    defensible_rate = mean(score == 1, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    jurisdiction_label = case_when(
      jurisdiction == "UK" ~ "United Kingdom",
      jurisdiction == "US" ~ "United States",
      jurisdiction == "NZ" ~ "New Zealand",
      TRUE ~ jurisdiction
    )
  )

fig_supp_jurisdiction <- ggplot(jurisdiction_perf,
                                 aes(x = reorder(jurisdiction_label, mean_score), y = mean_score)) +
  geom_col(aes(fill = mean_score), alpha = 0.9, width = 0.7) +
  geom_text(aes(label = sprintf("%.2f\n(n=%d)", mean_score, n_cases)),
            hjust = -0.1, size = 3.5) +
  coord_flip() +
  scale_fill_gradient(low = "#f39c12", high = "#2ecc71", guide = "none") +
  scale_y_continuous(limits = c(0, 0.8)) +
  labs(
    title = "Legal Defensibility by Jurisdiction",
    subtitle = "Mean defensibility score (0-1 scale) across common-law jurisdictions",
    x = NULL,
    y = "Mean Defensibility Score (0-1)"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 12),
    plot.subtitle = element_text(color = "gray40", size = 10)
  )

ggsave("output/fig_supp_jurisdiction.pdf", fig_supp_jurisdiction, width = 8, height = 5)

# ============================================
# SUPPLEMENTARY FIGURE 3: Performance by Medical Specialty
# ============================================

specialty_perf <- runs %>%
  filter(!is.na(specialty) & specialty != "unknown") %>%
  group_by(specialty) %>%
  summarise(
    n_cases = n_distinct(case_id),
    n_runs = n(),
    mean_score = mean(score, na.rm = TRUE),
    defensible_rate = mean(score == 1, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  filter(n_runs >= 10) %>%
  arrange(desc(mean_score))

fig_supp_specialty <- ggplot(specialty_perf %>% head(12),
                              aes(x = reorder(specialty, mean_score), y = mean_score)) +
  geom_point(aes(size = n_runs, color = mean_score), alpha = 0.8) +
  geom_segment(aes(xend = specialty, y = 0, yend = mean_score, color = mean_score),
               alpha = 0.6, linewidth = 1) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "gray40") +
  geom_text(aes(label = sprintf("%.2f", mean_score)),
            hjust = -0.3, size = 3) +
  coord_flip() +
  scale_color_gradient2(low = "#e74c3c", mid = "#f39c12", high = "#2ecc71",
                        midpoint = 0.5, limits = c(0, 1), guide = "none") +
  scale_size_continuous(range = c(3, 10), name = "N Runs") +
  labs(
    title = "Legal Defensibility by Medical Specialty",
    subtitle = "Top 12 specialties (minimum 10 runs) | Labels show mean score",
    x = NULL,
    y = "Mean Score (0-1)"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 12),
    plot.subtitle = element_text(color = "gray40", size = 10),
    legend.position = "right"
  ) +
  expand_limits(y = 1.05)

ggsave("output/fig_supp_specialty.pdf", fig_supp_specialty, width = 9, height = 6)

# ============================================
# SUPPLEMENTARY FIGURE 4: Length vs Defensibility (Model-Level)
# ============================================

# Calculate model-level correlation
length_def_cor_model <- cor(llm_summary$mean_length, llm_summary$defensible_rate)

fig_supp_length <- ggplot(llm_summary, aes(x = mean_length, y = defensible_rate)) +
  geom_point(aes(size = n, color = llm_display), alpha = 0.8) +
  geom_smooth(method = "lm", se = TRUE, color = "gray40", linetype = "dashed", linewidth = 0.8) +
  geom_text_repel(aes(label = llm_display), size = 3, box.padding = 0.4, point.padding = 0.3,
                  max.overlaps = 15, seed = 42, segment.color = "gray50") +
  scale_size_continuous(range = c(4, 12), name = "N Runs") +
  scale_color_manual(values = llm_colors, guide = "none") +
  scale_x_continuous(labels = scales::comma_format()) +
  scale_y_continuous(labels = scales::percent_format(), limits = c(0.2, 0.8)) +
  labs(
    title = "Response Length vs. Defensibility (Model-Level)",
    subtitle = sprintf("Pearson r = %.2f | Each point represents one LLM", length_def_cor_model),
    x = "Mean Recommendation Length (characters)",
    y = "Defensibility Rate (% scoring 1.0)"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 12),
    plot.subtitle = element_text(color = "gray40", size = 10),
    legend.position = "right"
  )

ggsave("output/fig_supp_length.pdf", fig_supp_length, width = 9, height = 6)

# ============================================
# SUPPLEMENTARY FIGURE 5: Clinical Reasoning vs Defensibility
# ============================================

# Calculate correlations
reasoning_def_cor_model <- cor(llm_summary$mean_reasoning, llm_summary$defensible_rate, use = "complete.obs")
reasoning_def_cor_indiv <- cor(runs$reasoning_quality_score, runs$score, use = "complete.obs")

fig_supp_reasoning <- ggplot(llm_summary, aes(x = mean_reasoning, y = defensible_rate)) +
  geom_point(aes(size = n, color = llm_display), alpha = 0.8) +
  geom_smooth(method = "lm", se = TRUE, color = "gray40", linetype = "dashed", linewidth = 0.8) +
  geom_text_repel(aes(label = llm_display), size = 3, box.padding = 0.4, point.padding = 0.3,
                  max.overlaps = 15, seed = 42, segment.color = "gray50") +
  scale_size_continuous(range = c(4, 12), name = "N Runs") +
  scale_color_manual(values = llm_colors, guide = "none") +
  scale_y_continuous(labels = scales::percent_format(), limits = c(0.2, 0.8)) +
  labs(
    title = "Clinical Reasoning Quality vs. Defensibility (Model-Level)",
    subtitle = sprintf("Model-level r = %.2f | Individual-level r = %.2f",
                       reasoning_def_cor_model, reasoning_def_cor_indiv),
    x = "Mean Clinical Reasoning Quality Score (0-1)",
    y = "Defensibility Rate (% scoring 1.0)"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 12),
    plot.subtitle = element_text(color = "gray40", size = 10),
    legend.position = "right"
  )

ggsave("output/fig_supp_reasoning.pdf", fig_supp_reasoning, width = 9, height = 6)

# ============================================
# SUPPLEMENTARY FIGURE: Reasoning Quality by Model (bar chart)
# ============================================

reasoning_by_model <- runs %>%
  filter(!is.na(reasoning_quality_score)) %>%
  group_by(llm_display) %>%
  summarise(
    mean_reasoning = mean(reasoning_quality_score, na.rm = TRUE),
    mean_score = mean(score, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  ) %>%
  mutate(llm_display = fct_reorder(llm_display, mean_reasoning))

fig_supp_reasoning_bars <- ggplot(reasoning_by_model, aes(x = llm_display, y = mean_reasoning)) +
  geom_col(aes(fill = mean_score), alpha = 0.9, width = 0.7) +
  geom_text(aes(label = sprintf("%.0f%%", mean_reasoning * 100)), hjust = -0.1, size = 3.5) +
  coord_flip() +
  scale_fill_gradient(low = "#e74c3c", high = "#2ecc71", name = "Defensibility") +
  labs(
    title = "Clinical Reasoning Quality by Model",
    subtitle = "Bar color indicates mean defensibility score",
    x = NULL,
    y = "Mean Reasoning Quality Score (0-1)"
  ) +
  expand_limits(y = 1.05) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 12),
    plot.subtitle = element_text(color = "gray40", size = 10),
    legend.position = "right"
  )

ggsave("output/fig_supp_reasoning_bars.pdf", fig_supp_reasoning_bars, width = 9, height = 6)

# ============================================
# SUPPLEMENTARY FIGURE 6: Test-Retest Reliability
# ============================================

# Find repeated runs (same case + same LLM)
# Use runs_all which includes all runs (including duplicates)
runs_for_retest <- runs_all

# Identify repeated case-LLM pairs
repeated_pairs <- runs_for_retest %>%
  group_by(case_id, llm_display) %>%
  filter(n() >= 2) %>%
  arrange(started_at) %>%
  mutate(run_num = row_number()) %>%
  ungroup()

if (nrow(repeated_pairs) > 0) {
  # Create concordance data
  concordance_data <- repeated_pairs %>%
    filter(run_num <= 2) %>%
    select(case_id, llm_display, run_num, score) %>%
    pivot_wider(names_from = run_num, values_from = score, names_prefix = "run_") %>%
    filter(!is.na(run_1) & !is.na(run_2))

  if (nrow(concordance_data) >= 5) {
    # Calculate metrics
    exact_agreement <- mean(concordance_data$run_1 == concordance_data$run_2)
    pearson_r <- cor(concordance_data$run_1, concordance_data$run_2)

    # Cohen's Kappa calculation
    run1_cat <- factor(concordance_data$run_1, levels = c(0, 0.5, 1))
    run2_cat <- factor(concordance_data$run_2, levels = c(0, 0.5, 1))
    confusion <- table(run1_cat, run2_cat)
    n_total <- sum(confusion)
    p_o <- sum(diag(confusion)) / n_total
    row_marginals <- rowSums(confusion) / n_total
    col_marginals <- colSums(confusion) / n_total
    p_e <- sum(row_marginals * col_marginals)
    cohens_kappa <- (p_o - p_e) / (1 - p_e)

    # Panel A: 2D density/count heatmap (since scores are only 0, 0.5, 1)
    count_matrix <- concordance_data %>%
      mutate(
        run_1_label = factor(run_1, levels = c(0, 0.5, 1), labels = c("0", "0.5", "1")),
        run_2_label = factor(run_2, levels = c(0, 0.5, 1), labels = c("0", "0.5", "1"))
      ) %>%
      count(run_1_label, run_2_label, name = "count") %>%
      mutate(
        is_agreement = run_1_label == run_2_label,
        pct = count / sum(count) * 100
      )

    fig_supp_retest <- ggplot(count_matrix, aes(x = run_1_label, y = run_2_label)) +
      geom_tile(aes(fill = count), color = "white", linewidth = 1.5) +
      geom_text(aes(label = sprintf("%d\n(%.0f%%)", count, pct),
                    color = is_agreement), size = 5, fontface = "bold") +
      scale_fill_gradient(low = "white", high = "#3498db", name = "Count") +
      scale_color_manual(values = c("TRUE" = "#27ae60", "FALSE" = "#e74c3c"), guide = "none") +
      labs(
        title = "Test-Retest Score Agreement",
        subtitle = sprintf("n = %d pairs | Exact agreement = %.0f%% | Cohen's kappa = %.2f | Green = agreement, Red = disagreement",
                           nrow(concordance_data), exact_agreement * 100, cohens_kappa),
        x = "First Run Score",
        y = "Second Run Score"
      ) +
      theme_minimal(base_size = 12) +
      theme(
        plot.title = element_text(face = "bold", size = 14),
        plot.subtitle = element_text(color = "gray40", size = 10),
        legend.position = "right",
        panel.grid = element_blank()
      ) +
      coord_fixed()

    ggsave("output/fig_supp_retest.pdf", fig_supp_retest, width = 7, height = 6)
  }
}

# ============================================
# SUPPLEMENTARY FIGURE 7: Readability Metrics Correlation Matrix
# ============================================

# Calculate word count for correlation matrix
runs <- runs %>%
  mutate(word_count = recommendation_length / 5)

# Create correlation matrix for ALL readability metrics including verbosity
readability_metrics_full <- runs %>%
  select(word_count, flesch_kincaid_grade, smog_index,
         transformer_readability_score, semantic_coherence_local, semantic_coherence_global) %>%
  filter(complete.cases(.))

# Rename for display
names(readability_metrics_full) <- c("Word Count", "Flesch-Kincaid",
                                      "SMOG", "Transformer", "Coherence (Local)", "Coherence (Global)")

cor_matrix_full <- cor(readability_metrics_full)

# Convert to long format for ggplot
cor_long <- cor_matrix_full %>%
  as.data.frame() %>%
  mutate(metric1 = rownames(.)) %>%
  pivot_longer(-metric1, names_to = "metric2", values_to = "correlation") %>%
  mutate(
    metric1 = factor(metric1, levels = colnames(cor_matrix_full)),
    metric2 = factor(metric2, levels = rev(colnames(cor_matrix_full)))
  )

# Correlation matrix heatmap
fig_supp_readability_cor <- ggplot(cor_long, aes(x = metric1, y = metric2, fill = correlation)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.2f", correlation)),
            size = 3, color = ifelse(abs(cor_long$correlation) > 0.5, "white", "black")) +
  scale_fill_gradient2(low = "#e74c3c", mid = "white", high = "#2ecc71",
                       midpoint = 0, limits = c(-1, 1), name = "r") +
  labs(
    title = "Readability Metrics Correlation Matrix",
    subtitle = "Showing relationships between response characteristics",
    x = NULL,
    y = NULL
  ) +
  theme_minimal(base_size = 10) +
  theme(
    plot.title = element_text(face = "bold", size = 12),
    plot.subtitle = element_text(color = "gray40", size = 10),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 9),
    axis.text.y = element_text(size = 9),
    panel.grid = element_blank(),
    legend.position = "right"
  ) +
  coord_fixed()

ggsave("output/fig_supp_readability_cor.pdf", fig_supp_readability_cor, width = 10, height = 8)

# ============================================
# SUPPLEMENTARY FIGURE 8: Score Distribution by LLM (Stacked Bar)
# ============================================

# Calculate proportions at each score level
score_dist <- runs %>%
  group_by(llm_display) %>%
  summarise(
    n = n(),
    prop_defensible = mean(score == 1, na.rm = TRUE),
    prop_partial = mean(score == 0.5, na.rm = TRUE),
    prop_liability = mean(score == 0, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  pivot_longer(cols = starts_with("prop_"),
               names_to = "score_level",
               values_to = "proportion",
               names_prefix = "prop_") %>%
  mutate(
    score_level = factor(score_level,
                         levels = c("liability", "partial", "defensible"),
                         labels = c("Liability (0)", "Partial (0.5)", "Defensible (1)")),
    llm_display = fct_reorder(llm_display, ifelse(score_level == "Defensible (1)", proportion, 0), .fun = max)
  )

fig_supp_score_dist <- ggplot(score_dist, aes(x = llm_display, y = proportion, fill = score_level)) +
  geom_col(position = "stack", alpha = 0.9, width = 0.7) +
  coord_flip() +
  scale_fill_manual(values = c("Liability (0)" = "#e74c3c",
                               "Partial (0.5)" = "#f39c12",
                               "Defensible (1)" = "#2ecc71"),
                    name = "Score") +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(
    title = "Defensibility Score Distribution by Model",
    subtitle = "Proportion of responses at each score level (0, 0.5, 1.0)",
    x = NULL,
    y = "Proportion of Responses"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 12),
    plot.subtitle = element_text(color = "gray40", size = 10),
    legend.position = "right"
  )

ggsave("output/fig_supp_score_dist.pdf", fig_supp_score_dist, width = 9, height = 6)

# ============================================
# SUPPLEMENTARY FIGURE 9: Case Difficulty Distribution
# ============================================

# Calculate how many models scored >0 for each case
# NA scores are treated as 0 (non-defensible)
# The "0" bar shows cases where no model scored >0 (including the 33 excluded cases)

case_difficulty_full <- runs_first %>%
  # For each run, treat NA (score_valid=FALSE) as score=0
  mutate(score_for_difficulty = ifelse(score_valid, score, 0)) %>%
  group_by(case_id) %>%
  summarise(
    n_models = n_distinct(llm_display),
    # Count models with score > 0 (either 0.5 or 1.0)
    n_above_zero = sum(score_for_difficulty > 0, na.rm = TRUE),
    .groups = "drop"
  )

# Report counts
n_zero_cases <- sum(case_difficulty_full$n_above_zero == 0)
cat(sprintf("Case difficulty: %d cases with 0 models scoring >0\n", n_zero_cases))
cat(sprintf("Total cases in figure: %d\n", nrow(case_difficulty_full)))

fig_supp_difficulty <- ggplot(case_difficulty_full, aes(x = factor(n_above_zero))) +
  geom_bar(aes(fill = n_above_zero), alpha = 0.9) +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.5, size = 3.5) +
  scale_fill_gradient(low = "#e74c3c", high = "#2ecc71", guide = "none") +
  labs(
    title = "Case Difficulty Distribution",
    subtitle = sprintf("Number of models scoring above zero per case (n=%d; NA treated as 0)", nrow(case_difficulty_full)),
    x = "Number of Models with Score > 0",
    y = "Number of Cases"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 12),
    plot.subtitle = element_text(color = "gray40", size = 10)
  )

ggsave("output/fig_supp_difficulty.pdf", fig_supp_difficulty, width = 8, height = 5)

# ============================================
# SUPPLEMENTARY FIGURE 10: Most Common Procedures
# ============================================

# Load procedure data and get most common
procedures_raw <- read_csv("../results/cost_summary.csv", show_col_types = FALSE)

# Need to get individual procedure data - check if we have it
procedure_details_file <- "../results/procedure_details.csv"
if (file.exists(procedure_details_file)) {
  procedure_details <- read_csv(procedure_details_file, show_col_types = FALSE)

  top_procedures <- procedure_details %>%
    count(procedure_name, sort = TRUE) %>%
    head(20) %>%
    left_join(
      procedure_details %>%
        group_by(procedure_name) %>%
        summarise(mean_cost = mean(cost, na.rm = TRUE), .groups = "drop"),
      by = "procedure_name"
    ) %>%
    mutate(procedure_name = fct_reorder(procedure_name, n))

  fig_supp_procedures <- ggplot(top_procedures, aes(x = procedure_name, y = n, fill = mean_cost)) +
    geom_col(alpha = 0.9, width = 0.7) +
    coord_flip() +
    scale_fill_gradient(low = "#74b9ff", high = "#0984e3", name = "Mean Cost ($)") +
    labs(
      title = "Most Commonly Recommended Procedures",
      subtitle = "Top 20 procedures across all LLM recommendations",
      x = NULL,
      y = "Frequency"
    ) +
    theme_minimal(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold", size = 12),
      plot.subtitle = element_text(color = "gray40", size = 10),
      legend.position = "right"
    )

  ggsave("output/fig_supp_procedures.pdf", fig_supp_procedures, width = 10, height = 7)
} else {
  cat("Note: procedure_details.csv not found, skipping common procedures figure\n")
}

# ============================================
# SUPPLEMENTARY FIGURE 11: Cost by Defensibility Stratum
# ============================================

# Show cost distributions within each defensibility level
# Three boxplots side-by-side for each model (not flipped, y-axis = cost)
# Count outliers for caption (now at $5K limit)
n_outliers_strat <- runs %>% filter(total_cost > 5000) %>% nrow()

# Order models by mean score for consistent display
model_order <- llm_summary %>% arrange(desc(mean_score)) %>% pull(llm_display)

fig_supp_cost_strat <- runs %>%
  mutate(
    score_label = factor(score, levels = c(0, 0.5, 1),
                         labels = c("0", "0.5", "1")),
    llm_display = factor(llm_display, levels = model_order)
  ) %>%
  ggplot(aes(x = llm_display, y = total_cost, fill = score_label)) +
  geom_boxplot(alpha = 0.8, outlier.size = 0.5, position = position_dodge(width = 0.8)) +
  scale_fill_manual(values = c("0" = "#e74c3c", "0.5" = "#f39c12", "1" = "#2ecc71"),
                    name = "Score\n(y-axis limited\nto $5K)") +
  scale_y_continuous(labels = scales::dollar_format(), limits = c(0, 5000)) +
  labs(
    title = "Cost Distribution Stratified by Defensibility Score",
    subtitle = sprintf("Within each model, three boxplots show cost by score (0, 0.5, 1). Y-axis limited to $5K; %d runs above limit omitted.", n_outliers_strat),
    x = NULL,
    y = "Total Cost ($)"
  ) +
  theme_minimal(base_size = 10) +
  theme(
    plot.title = element_text(face = "bold", size = 12),
    plot.subtitle = element_text(color = "gray40", size = 10),
    axis.text.x = element_text(angle = 30, hjust = 1, size = 9),
    legend.position = "right"
  )

ggsave("output/fig_supp_cost_strat.pdf", fig_supp_cost_strat, width = 12, height = 6)

# ============================================
# STRATIFIED ANALYSIS: Within-stratum effect sizes & within-model correlations
# (Addresses reviewer concern about capability-cost confound)
# ============================================

cat("\n=== Stratified Cost-Defensibility Analysis ===\n")

# 1. Overall eta-squared: how much cost variance is explained by model identity
overall_aov <- aov(total_cost ~ llm_display, data = runs)
overall_ss <- summary(overall_aov)[[1]]
overall_eta2 <- overall_ss$`Sum Sq`[1] / sum(overall_ss$`Sum Sq`)
cat(sprintf("Overall eta-squared (model -> cost): %.3f\n", overall_eta2))

# 2. Within-stratum eta-squared
strata_eta2 <- runs %>%
  filter(!is.na(score)) %>%
  group_by(score) %>%
  summarise(
    n = n(),
    eta2 = {
      if (n_distinct(llm_display) > 1 & n() > n_distinct(llm_display)) {
        a <- aov(total_cost ~ llm_display)
        ss <- summary(a)[[1]]
        ss$`Sum Sq`[1] / sum(ss$`Sum Sq`)
      } else { NA_real_ }
    },
    f_stat = {
      if (n_distinct(llm_display) > 1 & n() > n_distinct(llm_display)) {
        a <- aov(total_cost ~ llm_display)
        ss <- summary(a)[[1]]
        ss$`F value`[1]
      } else { NA_real_ }
    },
    p_value = {
      if (n_distinct(llm_display) > 1 & n() > n_distinct(llm_display)) {
        a <- aov(total_cost ~ llm_display)
        ss <- summary(a)[[1]]
        ss$`Pr(>F)`[1]
      } else { NA_real_ }
    },
    .groups = "drop"
  )

cat("\nWithin-stratum eta-squared (model -> cost):\n")
for (i in seq_len(nrow(strata_eta2))) {
  cat(sprintf("  Score %.1f: eta2=%.3f, F=%.1f, p=%.4f, n=%d\n",
              strata_eta2$score[i], strata_eta2$eta2[i],
              strata_eta2$f_stat[i], strata_eta2$p_value[i], strata_eta2$n[i]))
}

# 3. Within-model correlations: cost vs score for each model's own cases
within_model_cors <- runs %>%
  filter(!is.na(score) & !is.na(total_cost)) %>%
  group_by(llm_display) %>%
  summarise(
    n = n(),
    r = cor(total_cost, score, method = "spearman"),
    p_value = {
      if (n() > 3) {
        cor.test(total_cost, score, method = "spearman")$p.value
      } else { NA_real_ }
    },
    mean_cost_score0 = mean(total_cost[score == 0], na.rm = TRUE),
    mean_cost_score1 = mean(total_cost[score == 1], na.rm = TRUE),
    .groups = "drop"
  )

cat("\nWithin-model cost-defensibility correlations (Spearman):\n")
for (i in seq_len(nrow(within_model_cors))) {
  cat(sprintf("  %s: r=%.3f, p=%.4f (n=%d)\n",
              within_model_cors$llm_display[i], within_model_cors$r[i],
              within_model_cors$p_value[i], within_model_cors$n[i]))
}
mean_within_r <- mean(within_model_cors$r, na.rm = TRUE)
n_sig <- sum(within_model_cors$p_value < 0.05, na.rm = TRUE)
cat(sprintf("\nMean within-model r: %.3f (%d/%d significant at p<0.05)\n",
            mean_within_r, n_sig, nrow(within_model_cors)))

# ============================================
# ENHANCED FIGURE: Stratified Cost Analysis (3 panels)
# ============================================

# Panel A: existing stratified boxplots (reuse fig_supp_cost_strat)
panel_a <- fig_supp_cost_strat +
  labs(title = "A. Cost by Defensibility Stratum") +
  theme(plot.title = element_text(face = "bold", size = 11))

# Panel B: Eta-squared comparison
eta2_df <- bind_rows(
  tibble(stratum = "Overall", eta2 = overall_eta2),
  strata_eta2 %>% mutate(stratum = paste0("Score = ", score)) %>% select(stratum, eta2)
)
eta2_df$stratum <- factor(eta2_df$stratum, levels = c("Overall", "Score = 0", "Score = 0.5", "Score = 1"))

panel_b <- ggplot(eta2_df, aes(x = stratum, y = eta2, fill = stratum)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = sprintf("%.2f", eta2)), vjust = -0.3, size = 3.5) +
  scale_fill_manual(values = c("Overall" = "#34495e", "Score = 0" = "#e74c3c",
                                "Score = 0.5" = "#f39c12", "Score = 1" = "#2ecc71")) +
  scale_y_continuous(limits = c(0, max(eta2_df$eta2, na.rm = TRUE) * 1.2)) +
  labs(
    title = expression(bold("B. Effect of Model Identity on Cost ("*eta^2*")")),
    subtitle = "Lower = model identity explains less cost variance",
    x = NULL, y = expression(eta^2)
  ) +
  theme_minimal(base_size = 10) +
  theme(
    plot.title = element_text(face = "bold", size = 11),
    plot.subtitle = element_text(color = "gray40", size = 9),
    legend.position = "none"
  )

# Panel C: Within-model correlation forest plot
within_model_cors <- within_model_cors %>%
  mutate(llm_display = factor(llm_display, levels = llm_display[order(r)]))

# Compute CIs via Fisher z-transform
within_model_cors <- within_model_cors %>%
  mutate(
    z = atanh(r),
    se = 1 / sqrt(n - 3),
    r_lo = tanh(z - 1.96 * se),
    r_hi = tanh(z + 1.96 * se)
  )

panel_c <- ggplot(within_model_cors, aes(x = r, y = llm_display)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray60") +
  geom_errorbar(aes(xmin = r_lo, xmax = r_hi), width = 0.2, color = "gray40", orientation = "y") +
  geom_point(aes(color = r > 0), size = 3) +
  geom_vline(xintercept = mean_within_r, linetype = "dotted", color = "#2c3e50", linewidth = 0.7) +
  annotate("text", x = mean_within_r, y = 0.5, label = sprintf("mean r = %.2f", mean_within_r),
           hjust = -0.1, size = 3, color = "#2c3e50") +
  scale_color_manual(values = c("TRUE" = "#2ecc71", "FALSE" = "#e74c3c"), guide = "none") +
  labs(
    title = "C. Within-Model Cost-Defensibility Correlation",
    subtitle = "Each point: Spearman r across a single model's cases",
    x = "Spearman r (cost vs. defensibility score)", y = NULL
  ) +
  theme_minimal(base_size = 10) +
  theme(
    plot.title = element_text(face = "bold", size = 11),
    plot.subtitle = element_text(color = "gray40", size = 9)
  )

# Combine into 3-panel figure
fig_stratified_enhanced <- panel_a / (panel_b | panel_c) +
  plot_annotation(
    title = "Stratified Cost-Defensibility Analysis",
    subtitle = "Disentangling model capability from defensive testing patterns",
    theme = theme(
      plot.title = element_text(face = "bold", size = 14),
      plot.subtitle = element_text(color = "gray40", size = 11)
    )
  )

ggsave("output/fig_stratified_enhanced.pdf", fig_stratified_enhanced, width = 14, height = 12)
cat("Saved: output/fig_stratified_enhanced.pdf\n")

# ============================================
# SUPPLEMENTARY FIGURE: Evaluator Agreement (4x4 heatmap including Human)
# ============================================

# Calculate pairwise agreement between the three LLM evaluators
evaluator_agreement <- runs %>%
  filter(!is.na(claude_score) & !is.na(grok_score) & !is.na(gpt5_score)) %>%
  summarise(
    claude_grok = mean(claude_score == grok_score),
    claude_gpt5 = mean(claude_score == gpt5_score),
    grok_gpt5 = mean(grok_score == gpt5_score),
    all_agree = mean(claude_score == grok_score & grok_score == gpt5_score),
    n = n()
  )

# Load human validation data and calculate human-LLM agreement
# Human scored a subset of 121 cases
human_validation <- read_csv("../results/combined_validation.csv", show_col_types = FALSE) %>%
  filter(!is.na(C1_Human)) %>%
  # Convert scores to same scale (0, 1, 2)
  mutate(human_score = C1_Human)

# Calculate human-LLM pairwise agreement (binary: 0 vs 2, excluding 1)
human_binary <- human_validation %>%
  filter(majority_score %in% c(0, 2) & human_score %in% c(0, 2))

human_majority_agree <- mean(human_binary$human_score == human_binary$majority_score)
human_claude_agree <- human_validation %>%
  filter(claude_score %in% c(0, 2) & human_score %in% c(0, 2)) %>%
  summarise(agree = mean(claude_score == human_score)) %>% pull(agree)
human_grok_agree <- human_validation %>%
  filter(grok_score %in% c(0, 2) & human_score %in% c(0, 2)) %>%
  summarise(agree = mean(grok_score == human_score)) %>% pull(agree)
human_gpt5_agree <- human_validation %>%
  filter(gpt5_score %in% c(0, 2) & human_score %in% c(0, 2)) %>%
  summarise(agree = mean(gpt5_score == human_score)) %>% pull(agree)

n_human <- nrow(human_validation)

cat(sprintf("\n=== Human Validation Agreement (n=%d, binary 0/2 only) ===\n", n_human))
cat(sprintf("Human-Majority: %.1f%%\n", human_majority_agree * 100))
cat(sprintf("Human-Claude: %.1f%%\n", human_claude_agree * 100))
cat(sprintf("Human-Grok: %.1f%%\n", human_grok_agree * 100))
cat(sprintf("Human-GPT5.2: %.1f%%\n", human_gpt5_agree * 100))

# Calculate LLM-Majority agreement (each evaluator vs majority vote)
claude_majority_agree <- runs %>%
  filter(!is.na(claude_score) & !is.na(majority_score)) %>%
  filter(claude_score %in% c(0, 2) & majority_score %in% c(0, 2)) %>%
  summarise(agree = mean(claude_score == majority_score)) %>% pull(agree)
grok_majority_agree <- runs %>%
  filter(!is.na(grok_score) & !is.na(majority_score)) %>%
  filter(grok_score %in% c(0, 2) & majority_score %in% c(0, 2)) %>%
  summarise(agree = mean(grok_score == majority_score)) %>% pull(agree)
gpt5_majority_agree <- runs %>%
  filter(!is.na(gpt5_score) & !is.na(majority_score)) %>%
  filter(gpt5_score %in% c(0, 2) & majority_score %in% c(0, 2)) %>%
  summarise(agree = mean(gpt5_score == majority_score)) %>% pull(agree)

cat(sprintf("\n=== LLM-Majority Agreement (binary 0/2 only) ===\n"))
cat(sprintf("Claude-Majority: %.1f%%\n", claude_majority_agree * 100))
cat(sprintf("Grok-Majority: %.1f%%\n", grok_majority_agree * 100))
cat(sprintf("GPT5.2-Majority: %.1f%%\n", gpt5_majority_agree * 100))

# Create 5x5 agreement matrix including Human and Majority
# Note: Human agreement is calculated on subset of 121 cases (binary)
evaluators <- c("Claude", "Grok", "GPT-5.2", "Majority", "Human*")
agreement_values <- matrix(c(
  # Claude row
  1.0, evaluator_agreement$claude_grok, evaluator_agreement$claude_gpt5, claude_majority_agree, human_claude_agree,
  # Grok row
  evaluator_agreement$claude_grok, 1.0, evaluator_agreement$grok_gpt5, grok_majority_agree, human_grok_agree,
  # GPT-5.2 row
  evaluator_agreement$claude_gpt5, evaluator_agreement$grok_gpt5, 1.0, gpt5_majority_agree, human_gpt5_agree,
  # Majority row
  claude_majority_agree, grok_majority_agree, gpt5_majority_agree, 1.0, human_majority_agree,
  # Human row
  human_claude_agree, human_grok_agree, human_gpt5_agree, human_majority_agree, 1.0
), nrow = 5, byrow = TRUE)

agreement_matrix <- expand.grid(evaluator1 = evaluators, evaluator2 = evaluators) %>%
  mutate(
    agreement = as.vector(agreement_values),
    evaluator1 = factor(evaluator1, levels = evaluators),
    evaluator2 = factor(evaluator2, levels = rev(evaluators))
  )

fig_supp_evaluator_agreement <- ggplot(agreement_matrix, aes(x = evaluator1, y = evaluator2, fill = agreement)) +
  geom_tile(color = "white", linewidth = 1.5) +
  geom_text(aes(label = sprintf("%.0f%%", agreement * 100)),
            color = ifelse(agreement_matrix$agreement > 0.75, "white", "black"),
            size = 4, fontface = "bold") +
  scale_fill_gradient(low = "#f8d7da", high = "#28a745", limits = c(0.5, 1), name = "Agreement") +
  labs(
    title = "Evaluator Agreement Matrix",
    subtitle = sprintf("LLM evaluators (n=%d runs); *Human scored subset of %d cases (binary 0/1)",
                       evaluator_agreement$n, n_human),
    x = NULL,
    y = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(color = "gray40", size = 10),
    panel.grid = element_blank(),
    legend.position = "right"
  ) +
  coord_fixed()

ggsave("output/fig_supp_evaluator_agreement_matrix.pdf", fig_supp_evaluator_agreement, width = 9, height = 8)

cat(sprintf("\n=== Evaluator Agreement ===\n"))
cat(sprintf("Claude-Grok: %.1f%%\n", evaluator_agreement$claude_grok * 100))
cat(sprintf("Claude-GPT5.2: %.1f%%\n", evaluator_agreement$claude_gpt5 * 100))
cat(sprintf("Grok-GPT5.2: %.1f%%\n", evaluator_agreement$grok_gpt5 * 100))
cat(sprintf("All three agree: %.1f%%\n", evaluator_agreement$all_agree * 100))

# ============================================
# SUPPLEMENTARY FIGURE S7 Panel B: Minority Vote Analysis
# ============================================

# Load minority vote data from Python analysis
minority_data <- read.csv("../results/evaluator_minority_votes.csv")
minority_by_score <- read.csv("../results/evaluator_minority_by_score.csv")

# Clean up evaluator names for display
minority_data <- minority_data %>%
  mutate(evaluator_display = case_when(
    evaluator == "claude" ~ "Claude",
    evaluator == "grok" ~ "Grok",
    evaluator == "gpt5" ~ "GPT-5.2",
    TRUE ~ evaluator
  ))

minority_by_score <- minority_by_score %>%
  mutate(evaluator_display = case_when(
    evaluator == "claude" ~ "Claude",
    evaluator == "grok" ~ "Grok",
    evaluator == "gpt5" ~ "GPT-5.2",
    TRUE ~ evaluator
  ))

# Panel B: Bar chart of minority votes by evaluator
fig_minority_votes <- ggplot(minority_data, aes(x = reorder(evaluator_display, -minority_votes),
                                                  y = minority_votes, fill = evaluator_display)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = sprintf("%d\n(%.0f%%)", minority_votes, proportion * 100)),
            vjust = -0.3, size = 3.5) +
  scale_fill_manual(values = c("Claude" = "#E69F00", "Grok" = "#56B4E9", "GPT-5.2" = "#009E73")) +
  labs(
    title = "Minority Vote Frequency in 2-1 Splits",
    subtitle = "When evaluators disagree, which is most often in the minority?",
    x = "Evaluator",
    y = "Times in Minority"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(color = "gray40", size = 10),
    legend.position = "none"
  ) +
  ylim(0, max(minority_data$minority_votes) * 1.15)

# Panel C: Stacked bar showing what scores the minority evaluator gave
# Note: Internal scores are 0/1/2, displayed as 0/0.5/1 in paper
fig_minority_by_score <- ggplot(minority_by_score, aes(x = evaluator_display, y = count, fill = factor(minority_score))) +
  geom_col(position = "stack", width = 0.7) +
  scale_fill_manual(values = c("0" = "#d73027", "1" = "#fee08b", "2" = "#1a9850"),
                    labels = c("0" = "Score 0 (liability)", "1" = "Score 0.5 (partial)", "2" = "Score 1 (defensible)"),
                    name = "Minority Vote") +
  labs(
    title = "Minority Vote by Score Level",
    subtitle = "What score did the minority evaluator give?",
    x = "Evaluator",
    y = "Count"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(color = "gray40", size = 10),
    legend.position = "right"
  )

# Combine into S7 figure with panels A, B, C
library(patchwork)
fig_supp_evaluator_combined <- (fig_supp_evaluator_agreement + fig_minority_votes) / fig_minority_by_score +
  plot_annotation(tag_levels = 'A') +
  plot_layout(heights = c(1, 0.8))

ggsave("output/fig_supp_evaluator_agreement.pdf", fig_supp_evaluator_combined, width = 12, height = 10)

# ============================================
# SUPPLEMENTARY FIGURE: Average vs Majority Score Comparison
# ============================================

# Compare model rankings under majority vote vs average score
llm_summary_avg <- runs %>%
  group_by(llm_display) %>%
  summarise(
    mean_score_majority = mean(score, na.rm = TRUE),
    mean_score_avg = mean(score_avg, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_score_majority))

# Correlation between methods
majority_avg_cor <- cor(llm_summary_avg$mean_score_majority, llm_summary_avg$mean_score_avg)

# Create comparison bar chart
comparison_long <- llm_summary_avg %>%
  pivot_longer(cols = c(mean_score_majority, mean_score_avg),
               names_to = "method", values_to = "score") %>%
  mutate(
    method = ifelse(method == "mean_score_majority", "Majority Vote", "Average Score"),
    llm_display = fct_reorder(llm_display, score, .fun = max)
  )

fig_supp_majority_vs_avg <- ggplot(comparison_long, aes(x = llm_display, y = score, fill = method)) +
  geom_col(position = position_dodge(width = 0.8), alpha = 0.9, width = 0.7) +
  geom_text(aes(label = sprintf("%.2f", score)),
            position = position_dodge(width = 0.8), hjust = -0.1, size = 3) +
  coord_flip() +
  scale_fill_manual(values = c("Majority Vote" = "#2ecc71", "Average Score" = "#3498db"),
                    name = "Scoring Method") +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    title = "Majority Vote vs Average Score Comparison",
    subtitle = sprintf("Model defensibility scores under two scoring methods (r = %.2f)", majority_avg_cor),
    x = NULL,
    y = "Mean Defensibility Score (0-1)"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 12),
    plot.subtitle = element_text(color = "gray40", size = 10),
    legend.position = "bottom"
  )

ggsave("output/fig_supp_majority_vs_avg.pdf", fig_supp_majority_vs_avg, width = 10, height = 6)

cat(sprintf("\n=== Majority vs Average Score ===\n"))
cat(sprintf("Correlation between methods: r = %.3f\n", majority_avg_cor))

# ============================================
# SUPPLEMENTARY FIGURE: Self-Evaluation Bias Analysis
# ============================================

# Analyze whether models score themselves higher (exact self-match only)
# GPT-5.2 responses scored by GPT-5.2 evaluator vs. other evaluators
# Claude responses scored by Claude evaluator vs. other evaluators
# Grok responses scored by Grok evaluator vs. other evaluators

self_eval_analysis <- runs %>%
  filter(!is.na(claude_score) & !is.na(grok_score) & !is.na(gpt5_score)) %>%
  mutate(
    # For each model, compare self-score vs mean of other two evaluators
    gpt5_self = ifelse(llm_display == "GPT-5.2", gpt5_score / 2, NA),
    gpt5_others = ifelse(llm_display == "GPT-5.2", (claude_score + grok_score) / 4, NA),
    claude_self = ifelse(llm_display == "Claude Sonnet 4", claude_score / 2, NA),
    claude_others = ifelse(llm_display == "Claude Sonnet 4", (gpt5_score + grok_score) / 4, NA),
    grok_self = ifelse(llm_display == "Grok 4.1", grok_score / 2, NA),
    grok_others = ifelse(llm_display == "Grok 4.1", (claude_score + gpt5_score) / 4, NA)
  )

# Aggregate self-evaluation bias
self_eval_summary <- data.frame(
  model = c("GPT-5.2", "Claude Sonnet 4", "Grok 4.1"),
  self_score = c(
    mean(self_eval_analysis$gpt5_self, na.rm = TRUE),
    mean(self_eval_analysis$claude_self, na.rm = TRUE),
    mean(self_eval_analysis$grok_self, na.rm = TRUE)
  ),
  other_score = c(
    mean(self_eval_analysis$gpt5_others, na.rm = TRUE),
    mean(self_eval_analysis$claude_others, na.rm = TRUE),
    mean(self_eval_analysis$grok_others, na.rm = TRUE)
  ),
  n = c(
    sum(!is.na(self_eval_analysis$gpt5_self)),
    sum(!is.na(self_eval_analysis$claude_self)),
    sum(!is.na(self_eval_analysis$grok_self))
  )
) %>%
  mutate(
    bias = self_score - other_score,
    model = factor(model, levels = c("GPT-5.2", "Claude Sonnet 4", "Grok 4.1"))
  )

# Create comparison plot
self_eval_long <- self_eval_summary %>%
  pivot_longer(cols = c(self_score, other_score),
               names_to = "evaluator_type", values_to = "score") %>%
  mutate(
    evaluator_type = ifelse(evaluator_type == "self_score", "Self-Evaluation", "Other Evaluators")
  )

fig_supp_self_eval <- ggplot(self_eval_long, aes(x = model, y = score, fill = evaluator_type)) +
  geom_col(position = position_dodge(width = 0.7), alpha = 0.9, width = 0.6) +
  geom_text(aes(label = sprintf("%.2f", score)),
            position = position_dodge(width = 0.7), vjust = -0.5, size = 3.5) +
  geom_text(data = self_eval_summary,
            aes(x = model, y = max(self_score, other_score) + 0.12,
                label = sprintf("Δ = %+.2f", bias)),
            inherit.aes = FALSE, size = 3.5, fontface = "bold",
            color = ifelse(self_eval_summary$bias > 0, "#e74c3c", "#27ae60")) +
  scale_fill_manual(values = c("Self-Evaluation" = "#e74c3c", "Other Evaluators" = "#3498db"),
                    name = NULL) +
  scale_y_continuous(limits = c(0, 1.1)) +
  labs(
    title = "Self-Evaluation Bias Analysis",
    subtitle = "Do models score their own outputs higher? (Δ = self - others)",
    x = NULL,
    y = "Mean Score (0-1)"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 12),
    plot.subtitle = element_text(color = "gray40", size = 10),
    legend.position = "bottom"
  )

ggsave("output/fig_supp_self_eval.pdf", fig_supp_self_eval, width = 9, height = 6)

cat(sprintf("\n=== Self-Evaluation Bias ===\n"))
for (i in 1:nrow(self_eval_summary)) {
  row <- self_eval_summary[i, ]
  cat(sprintf("%s: self=%.2f, others=%.2f, bias=%+.2f (n=%d)\n",
              row$model, row$self_score, row$other_score, row$bias, row$n))
}

# ============================================
# SUPPLEMENTARY TABLE 4: Enhanced Model Performance Summary
# ============================================

supp_table4 <- llm_summary %>%
  arrange(desc(mean_score)) %>%
  select(
    Model = llm_display,
    N = n,
    `Mean Score` = mean_score,
    `Defensible %` = defensible_rate,
    `Partial %` = partial_rate,
    `Liability %` = liability_rate,
    `Reasoning` = mean_reasoning,
    `Length` = mean_length,
    `Readability` = mean_readability,
    `Mean Cost` = mean_cost,
    `Procedures` = mean_procedures,
    CPDR = cpdr
  ) %>%
  mutate(across(where(is.numeric), ~round(., 2)))

write_csv(supp_table4, "figures/supp_table4_model_summary.csv")
cat("\n=== Supplementary Table 4: Model Summary ===\n")
print(supp_table4)

# ============================================
# Print summary for paper
# ============================================

cat("\n=== Summary Statistics for Paper ===\n\n")
cat(sprintf("Total valid runs: %d\n", nrow(runs)))
cat(sprintf("Unique cases: %d\n", n_distinct(runs$case_id)))
cat(sprintf("LLMs evaluated: %d\n", n_distinct(runs$llm_display)))
cat(sprintf("Overall mean score: %.3f\n", mean(runs$score)))
cat(sprintf("Overall defensibility rate: %.1f%%\n", mean(runs$score == 1) * 100))
cat(sprintf("Overall partial rate: %.1f%%\n", mean(runs$score == 0.5) * 100))
cat(sprintf("Overall liability rate: %.1f%%\n", mean(runs$score == 0) * 100))
cat(sprintf("Overall mean length: %.0f chars\n", mean(runs$recommendation_length, na.rm = TRUE)))
cat(sprintf("Overall mean readability (transformer): %.1f\n", mean(runs$transformer_readability_score, na.rm = TRUE)))
cat(sprintf("Overall mean cost: $%.2f\n", mean(runs$total_cost, na.rm = TRUE)))
cat(sprintf("Overall mean procedures: %.1f\n", mean(runs$procedure_count, na.rm = TRUE)))

cat("\n--- Spearman Correlations ---\n")
cat(sprintf("Model-level length-score correlation: rho = %.3f\n",
            cor(llm_summary$mean_length, llm_summary$mean_score, method = "spearman")))
cat(sprintf("Model-level readability-score correlation: rho = %.3f\n",
            cor(llm_summary$mean_readability, llm_summary$mean_score, use = "complete.obs", method = "spearman")))
cat(sprintf("Model-level cost-score correlation: rho = %.3f\n",
            cor(llm_summary$mean_cost, llm_summary$mean_score, use = "complete.obs", method = "spearman")))
cat(sprintf("Model-level procedures-score correlation: rho = %.3f\n",
            cor(llm_summary$mean_procedures, llm_summary$mean_score, use = "complete.obs", method = "spearman")))
cat(sprintf("Individual-level length-score correlation: rho = %.3f\n",
            cor(runs$recommendation_length, runs$score, use = "complete.obs", method = "spearman")))
cat(sprintf("Individual-level cost-score correlation: rho = %.3f\n",
            cor(runs$total_cost, runs$score, use = "complete.obs", method = "spearman")))

cat("\n--- By Model (ordered by mean score) ---\n")
print(llm_summary %>%
        arrange(desc(mean_score)) %>%
        select(llm_display, n, mean_score, defensible_rate, mean_length, mean_reasoning, mean_readability, mean_cost, mean_procedures, cpdr) %>%
        mutate(across(where(is.numeric), ~round(., 3))))

# ============================================
# STATISTICAL TESTS FOR PAPER CLAIMS
# ============================================

cat("\n\n" %+% paste(rep("=", 60), collapse = "") %+% "\n")
cat("STATISTICAL TESTS FOR PAPER\n")
cat(paste(rep("=", 60), collapse = "") %+% "\n\n")

# Helper function for formatting p-values
format_p <- function(p) {
  if (p < 0.001) return("P < 0.001")
  if (p < 0.01) return(sprintf("P = %.3f", p))
  return(sprintf("P = %.2f", p))
}

# Helper for string concatenation
`%+%` <- function(a, b) paste0(a, b)

# --- 1. PAIRWISE MODEL COMPARISONS (Defensibility) ---
cat("=== 1. Pairwise Model Comparisons (Defensibility) ===\n")
cat("Using Mann-Whitney U tests (Wilcoxon rank-sum) for non-parametric comparisons\n\n")

# Get unique model names ordered by score
model_order <- llm_summary %>% arrange(desc(mean_score)) %>% pull(llm_display)

# Create pairwise comparison matrix
pairwise_results <- data.frame()
for (i in 1:(length(model_order)-1)) {
  for (j in (i+1):length(model_order)) {
    model1 <- model_order[i]
    model2 <- model_order[j]

    scores1 <- runs %>% filter(llm_display == model1) %>% pull(score)
    scores2 <- runs %>% filter(llm_display == model2) %>% pull(score)

    # Mann-Whitney U test
    test <- wilcox.test(scores1, scores2)

    # Effect size (rank-biserial correlation)
    n1 <- length(scores1)
    n2 <- length(scores2)
    U <- test$statistic
    r <- 1 - (2 * U) / (n1 * n2)  # rank-biserial correlation

    pairwise_results <- rbind(pairwise_results, data.frame(
      model1 = model1,
      model2 = model2,
      mean1 = mean(scores1),
      mean2 = mean(scores2),
      diff = mean(scores1) - mean(scores2),
      U = U,
      p_value = test$p.value,
      r_effect = r
    ))
  }
}

# Adjust for multiple comparisons (Bonferroni)
pairwise_results$p_adjusted <- p.adjust(pairwise_results$p_value, method = "bonferroni")

# Print key comparisons
cat("Key pairwise comparisons:\n")
key_comparisons <- pairwise_results %>%
  filter(model1 == "GPT-5.2" | model2 == "GPT-4o") %>%
  head(10)

for (i in 1:nrow(key_comparisons)) {
  row <- key_comparisons[i, ]
  sig <- ifelse(row$p_adjusted < 0.05, "*", "")
  cat(sprintf("  %s (%.2f) vs %s (%.2f): diff=%.2f, %s, r=%.2f%s\n",
              row$model1, row$mean1, row$model2, row$mean2, row$diff,
              format_p(row$p_adjusted), row$r_effect, sig))
}

# --- 2. KRUSKAL-WALLIS TEST (Overall model effect) ---
cat("\n=== 2. Overall Model Effect (Kruskal-Wallis) ===\n")
kw_test <- kruskal.test(score ~ llm_display, data = runs)
cat(sprintf("Kruskal-Wallis chi-squared = %.2f, df = %d, %s\n",
            kw_test$statistic, kw_test$parameter, format_p(kw_test$p.value)))

# --- 3. RESPONSE LENGTH COMPARISONS ---
cat("\n=== 3. Response Length Comparisons ===\n")

# GPT-5.2 vs GPT-4o length comparison
gpt5_length <- runs %>% filter(llm_display == "GPT-5.2") %>% pull(recommendation_length)
gpt4o_length <- runs %>% filter(llm_display == "GPT-4o") %>% pull(recommendation_length)

# t-test for length
length_ttest <- t.test(gpt5_length, gpt4o_length)
# Cohen's d
pooled_sd <- sqrt(((length(gpt5_length)-1)*sd(gpt5_length)^2 + (length(gpt4o_length)-1)*sd(gpt4o_length)^2) /
                    (length(gpt5_length) + length(gpt4o_length) - 2))
cohens_d_length <- (mean(gpt5_length) - mean(gpt4o_length)) / pooled_sd

cat(sprintf("GPT-5.2 vs GPT-4o response length:\n"))
cat(sprintf("  GPT-5.2: mean = %.0f chars (SD = %.0f)\n", mean(gpt5_length), sd(gpt5_length)))
cat(sprintf("  GPT-4o: mean = %.0f chars (SD = %.0f)\n", mean(gpt4o_length), sd(gpt4o_length)))
cat(sprintf("  Ratio: %.1fx longer\n", mean(gpt5_length) / mean(gpt4o_length)))
cat(sprintf("  t = %.2f, df = %.1f, %s\n", length_ttest$statistic, length_ttest$parameter, format_p(length_ttest$p.value)))
cat(sprintf("  Cohen's d = %.2f (large effect)\n", cohens_d_length))

# Llama 4 comparison
llama_length <- runs %>% filter(llm_display == "Llama 4 Maverick") %>% pull(recommendation_length)
length_ttest_llama <- t.test(gpt5_length, llama_length)
cohens_d_llama <- (mean(gpt5_length) - mean(llama_length)) /
  sqrt(((length(gpt5_length)-1)*sd(gpt5_length)^2 + (length(llama_length)-1)*sd(llama_length)^2) /
         (length(gpt5_length) + length(llama_length) - 2))
cat(sprintf("\nGPT-5.2 vs Llama 4 Maverick:\n"))
cat(sprintf("  Llama 4 Maverick: mean = %.0f chars (SD = %.0f)\n", mean(llama_length), sd(llama_length)))
cat(sprintf("  Ratio: %.1fx longer\n", mean(gpt5_length) / mean(llama_length)))
cat(sprintf("  t = %.2f, %s, Cohen's d = %.2f\n", length_ttest_llama$statistic, format_p(length_ttest_llama$p.value), cohens_d_llama))

# --- 4. READABILITY COMPARISONS ---
cat("\n=== 4. Readability Comparisons ===\n")

# GPT-5.2 vs Grok readability
gpt5_read <- runs %>% filter(llm_display == "GPT-5.2") %>% pull(transformer_readability_score) %>% na.omit()
grok_read <- runs %>% filter(llm_display == "Grok 4.1") %>% pull(transformer_readability_score) %>% na.omit()

read_ttest <- t.test(gpt5_read, grok_read)
cat(sprintf("GPT-5.2 vs Grok 4.1 readability:\n"))
cat(sprintf("  GPT-5.2: mean = %.2f (SD = %.2f)\n", mean(gpt5_read), sd(gpt5_read)))
cat(sprintf("  Grok 4.1: mean = %.2f (SD = %.2f)\n", mean(grok_read), sd(grok_read)))
cat(sprintf("  t = %.2f, %s\n", read_ttest$statistic, format_p(read_ttest$p.value)))

# --- 5. CORRELATION TESTS WITH P-VALUES ---
cat("\n=== 5. Correlation Tests with P-values ===\n")

# Individual-level correlations (Spearman - appropriate for ordinal scores)
cor_length <- cor.test(runs$recommendation_length, runs$score, method = "spearman")
cor_readability <- cor.test(runs$transformer_readability_score, runs$score, use = "complete.obs", method = "spearman")
cor_cost <- cor.test(runs$total_cost, runs$score, use = "complete.obs", method = "spearman")
cor_reasoning <- cor.test(runs$reasoning_quality_score, runs$score, use = "complete.obs", method = "spearman")

cat("Individual-level Spearman correlations (n = " %+% nrow(runs) %+% "):\n")
cat(sprintf("  Length-Defensibility: rho = %.3f, %s\n",
            cor_length$estimate, format_p(cor_length$p.value)))
cat(sprintf("  Readability-Defensibility: rho = %.3f, %s\n",
            cor_readability$estimate, format_p(cor_readability$p.value)))
cat(sprintf("  Cost-Defensibility: rho = %.3f, %s\n",
            cor_cost$estimate, format_p(cor_cost$p.value)))
cat(sprintf("  Reasoning-Defensibility: rho = %.3f, %s\n",
            cor_reasoning$estimate, format_p(cor_reasoning$p.value)))

# Model-level correlations (n=7, so report but note limited power)
cat("\nModel-level correlations (n = 7 models - interpret with caution due to small n):\n")
cor_length_model <- cor.test(llm_summary$mean_length, llm_summary$mean_score, method = "spearman")
cor_read_model <- cor.test(llm_summary$mean_readability, llm_summary$mean_score, use = "complete.obs", method = "spearman")
cor_cost_model <- cor.test(llm_summary$mean_cost, llm_summary$mean_score, use = "complete.obs", method = "spearman")
cor_reason_model <- cor.test(llm_summary$mean_reasoning, llm_summary$mean_score, use = "complete.obs", method = "spearman")

cat(sprintf("  Length-Defensibility: r = %.2f, %s\n", cor_length_model$estimate, format_p(cor_length_model$p.value)))
cat(sprintf("  Readability-Defensibility: r = %.2f, %s\n", cor_read_model$estimate, format_p(cor_read_model$p.value)))
cat(sprintf("  Cost-Defensibility: r = %.2f, %s\n", cor_cost_model$estimate, format_p(cor_cost_model$p.value)))
cat(sprintf("  Reasoning-Defensibility: r = %.2f, %s\n", cor_reason_model$estimate, format_p(cor_reason_model$p.value)))

# --- 6. JURISDICTION COMPARISONS ---
cat("\n=== 6. Jurisdiction Comparisons ===\n")

# Kruskal-Wallis test
jurisdiction_kw <- kruskal.test(score ~ jurisdiction, data = runs %>% filter(!is.na(jurisdiction)))
cat(sprintf("Kruskal-Wallis test: chi-squared = %.2f, df = %d, %s\n",
            jurisdiction_kw$statistic, jurisdiction_kw$parameter, format_p(jurisdiction_kw$p.value)))

# Pairwise Wilcoxon tests
uk_scores <- runs %>% filter(jurisdiction == "UK") %>% pull(score)
us_scores <- runs %>% filter(jurisdiction == "US") %>% pull(score)
nz_scores <- runs %>% filter(jurisdiction == "NZ") %>% pull(score)

uk_us <- wilcox.test(uk_scores, us_scores)
uk_nz <- wilcox.test(uk_scores, nz_scores)
us_nz <- wilcox.test(us_scores, nz_scores)

cat(sprintf("\nPairwise comparisons:\n"))
cat(sprintf("  UK (%.2f) vs US (%.2f): %s\n", mean(uk_scores), mean(us_scores), format_p(uk_us$p.value)))
cat(sprintf("  UK (%.2f) vs NZ (%.2f): %s\n", mean(uk_scores), mean(nz_scores), format_p(uk_nz$p.value)))
cat(sprintf("  US (%.2f) vs NZ (%.2f): %s\n", mean(us_scores), mean(nz_scores), format_p(us_nz$p.value)))

# --- 7. MALPRACTICE CATEGORY COMPARISONS ---
cat("\n=== 7. Malpractice Category Comparisons ===\n")

# Get top and bottom categories
top_category <- malpractice_perf %>% slice_max(mean_score, n = 1)
bottom_category <- malpractice_perf %>% slice_min(mean_score, n = 1)

top_scores <- runs %>% filter(malpractice_display == top_category$malpractice_display) %>% pull(score)
bottom_scores <- runs %>% filter(malpractice_display == bottom_category$malpractice_display) %>% pull(score)

mal_test <- wilcox.test(top_scores, bottom_scores)

cat(sprintf("Best category: %s (mean = %.2f, n = %d)\n",
            top_category$malpractice_display, top_category$mean_score, length(top_scores)))
cat(sprintf("Worst category: %s (mean = %.2f, n = %d)\n",
            bottom_category$malpractice_display, bottom_category$mean_score, length(bottom_scores)))
cat(sprintf("Comparison: %s\n", format_p(mal_test$p.value)))

# Overall category effect
mal_kw <- kruskal.test(score ~ malpractice_display, data = runs)
cat(sprintf("\nOverall category effect (Kruskal-Wallis): chi-squared = %.2f, df = %d, %s\n",
            mal_kw$statistic, mal_kw$parameter, format_p(mal_kw$p.value)))

# --- 8. COST COMPARISONS ---
cat("\n=== 8. Cost Comparisons ===\n")

gpt5_cost <- runs %>% filter(llm_display == "GPT-5.2") %>% pull(total_cost)
gpt4o_cost <- runs %>% filter(llm_display == "GPT-4o") %>% pull(total_cost)
gemini_cost <- runs %>% filter(llm_display == "Gemini 3 Pro") %>% pull(total_cost)
claude_cost <- runs %>% filter(llm_display == "Claude Sonnet 4") %>% pull(total_cost)

cost_test <- wilcox.test(gpt5_cost, gpt4o_cost)
cat(sprintf("GPT-5.2 ($%.0f) vs GPT-4o ($%.0f): %s\n",
            mean(gpt5_cost), mean(gpt4o_cost), format_p(cost_test$p.value)))

cost_test_gemini <- wilcox.test(gpt5_cost, gemini_cost)
cat(sprintf("GPT-5.2 ($%.0f) vs Gemini 3 Pro ($%.0f): %s\n",
            mean(gpt5_cost), mean(gemini_cost), format_p(cost_test_gemini$p.value)))

cost_test_claude <- wilcox.test(claude_cost, gemini_cost)
cat(sprintf("Claude Sonnet 4 ($%.0f) vs Gemini 3 Pro ($%.0f): %s\n",
            mean(claude_cost), mean(gemini_cost), format_p(cost_test_claude$p.value)))

# Procedure count comparison
gpt5_procs <- runs %>% filter(llm_display == "GPT-5.2") %>% pull(procedure_count)
gpt4o_procs <- runs %>% filter(llm_display == "GPT-4o") %>% pull(procedure_count)
proc_test <- wilcox.test(gpt5_procs, gpt4o_procs)
cat(sprintf("\nProcedure counts:\n"))
cat(sprintf("  GPT-5.2: mean = %.1f procedures\n", mean(gpt5_procs, na.rm = TRUE)))
cat(sprintf("  GPT-4o: mean = %.1f procedures\n", mean(gpt4o_procs, na.rm = TRUE)))
cat(sprintf("  Comparison: %s\n", format_p(proc_test$p.value)))

# --- 9. TEST-RETEST RELIABILITY ---
cat("\n=== 9. Test-Retest Reliability ===\n")

if (exists("concordance_data") && nrow(concordance_data) >= 5) {
  # Already calculated earlier, report with CI
  # Bootstrap CI for kappa
  set.seed(42)
  n_boot <- 1000
  kappa_boot <- numeric(n_boot)

  for (b in 1:n_boot) {
    boot_idx <- sample(1:nrow(concordance_data), replace = TRUE)
    boot_data <- concordance_data[boot_idx, ]

    run1_cat <- factor(boot_data$run_1, levels = c(0, 0.5, 1))
    run2_cat <- factor(boot_data$run_2, levels = c(0, 0.5, 1))
    confusion <- table(run1_cat, run2_cat)
    n_total <- sum(confusion)
    p_o <- sum(diag(confusion)) / n_total
    row_marginals <- rowSums(confusion) / n_total
    col_marginals <- colSums(confusion) / n_total
    p_e <- sum(row_marginals * col_marginals)
    kappa_boot[b] <- (p_o - p_e) / (1 - p_e)
  }

  kappa_ci <- quantile(kappa_boot, c(0.025, 0.975), na.rm = TRUE)
  cat(sprintf("Cohen's kappa = %.2f, 95%% CI [%.2f, %.2f]\n", cohens_kappa, kappa_ci[1], kappa_ci[2]))
  cat(sprintf("Exact agreement = %.1f%%, n = %d pairs\n", exact_agreement * 100, nrow(concordance_data)))
}

# --- 10. EFFECT SIZE SUMMARY ---
cat("\n=== 10. Effect Size Summary ===\n")
cat("Key effect sizes for paper:\n")
cat(sprintf("  GPT-5.2 vs GPT-4o length: Cohen's d = %.2f\n", cohens_d_length))
cat(sprintf("  Length-defensibility (individual): r = %.3f\n", cor_length$estimate))
cat(sprintf("  Length-defensibility (model-level): r = %.2f\n", cor_length_model$estimate))
cat(sprintf("  Cost-defensibility (model-level): r = %.2f\n", cor_cost_model$estimate))
cat(sprintf("  Reasoning-defensibility (model-level): r = %.2f\n", cor_reason_model$estimate))

# Save pairwise comparisons for reference
write_csv(pairwise_results, "figures/pairwise_model_comparisons.csv")

# ============================================
# SUPPLEMENTARY FIGURE 12: Excluded Cases Characterization
# ============================================

# Get characteristics of excluded cases vs included cases
# excluded_cases defined earlier in the script

excluded_case_data <- runs_first %>%
  mutate(is_excluded = case_id %in% excluded_cases) %>%
  distinct(case_id, .keep_all = TRUE)

# By malpractice type
mal_exclusion_rates <- excluded_case_data %>%
  group_by(malpractice_display) %>%
  summarise(
    n_total = n(),
    n_excluded = sum(is_excluded),
    exclusion_rate = mean(is_excluded) * 100,
    .groups = "drop"
  ) %>%
  filter(n_total >= 3) %>%
  arrange(desc(exclusion_rate))

fig_supp_excluded_mal <- ggplot(mal_exclusion_rates,
                                  aes(x = reorder(malpractice_display, exclusion_rate),
                                      y = exclusion_rate)) +
  geom_col(aes(fill = exclusion_rate), alpha = 0.9, width = 0.7) +
  geom_text(aes(label = sprintf("%.0f%% (%d/%d)", exclusion_rate, n_excluded, n_total)),
            hjust = -0.1, size = 3) +
  coord_flip() +
  scale_fill_gradient(low = "#2ecc71", high = "#e74c3c", guide = "none") +
  scale_y_continuous(limits = c(0, 110)) +
  labs(
    title = "Exclusion Rate by Malpractice Type",
    subtitle = "Percentage of cases excluded (all models scored 0 or NA)",
    x = NULL,
    y = "Exclusion Rate (%)"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 12),
    plot.subtitle = element_text(color = "gray40", size = 10)
  )

# By specialty
spec_exclusion_rates <- excluded_case_data %>%
  filter(!is.na(specialty) & specialty != "unknown") %>%
  group_by(specialty) %>%
  summarise(
    n_total = n(),
    n_excluded = sum(is_excluded),
    exclusion_rate = mean(is_excluded) * 100,
    .groups = "drop"
  ) %>%
  filter(n_total >= 3, exclusion_rate > 0) %>%
  arrange(desc(exclusion_rate)) %>%
  head(12)

fig_supp_excluded_spec <- ggplot(spec_exclusion_rates,
                                   aes(x = reorder(specialty, exclusion_rate),
                                       y = exclusion_rate)) +
  geom_col(aes(fill = exclusion_rate), alpha = 0.9, width = 0.7) +
  geom_text(aes(label = sprintf("%.0f%% (%d/%d)", exclusion_rate, n_excluded, n_total)),
            hjust = -0.1, size = 3) +
  coord_flip() +
  scale_fill_gradient(low = "#2ecc71", high = "#e74c3c", guide = "none") +
  scale_y_continuous(limits = c(0, 110)) +
  labs(
    title = "Exclusion Rate by Medical Specialty",
    subtitle = "Top 12 specialties with highest exclusion rates (min 3 cases)",
    x = NULL,
    y = "Exclusion Rate (%)"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 12),
    plot.subtitle = element_text(color = "gray40", size = 10)
  )

# Combine into one figure
fig_supp_excluded <- fig_supp_excluded_mal / fig_supp_excluded_spec +
  plot_annotation(
    title = "Characterization of Excluded Cases",
    subtitle = sprintf("Cases where all %d models scored 0 or NA (n=%d excluded of %d total)",
                       n_llms, length(excluded_cases), n_distinct(runs_first$case_id)),
    tag_levels = "A"
  ) &
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(color = "gray40", size = 11)
  )

ggsave("output/fig_supp_excluded.pdf", fig_supp_excluded, width = 10, height = 12)

cat(sprintf("\n=== Excluded Cases Analysis ===\n"))
cat(sprintf("Total cases: %d\n", n_distinct(runs_first$case_id)))
cat(sprintf("Excluded cases: %d (%.1f%%)\n", length(excluded_cases),
            length(excluded_cases) / n_distinct(runs_first$case_id) * 100))
cat("\nTop malpractice types by exclusion rate:\n")
print(mal_exclusion_rates %>% head(5))

cat("\nFigures saved to paper/figures/\n")

# Write stats to file for easy reference in paper
stats_file <- "STATS_SUMMARY.txt"
sink(stats_file)
cat("=== CLAD Paper Statistics Summary ===\n")
cat(sprintf("Generated: %s\n\n", Sys.time()))

cat("--- Dataset ---\n")
cat(sprintf("Total valid runs: %d\n", nrow(runs)))
cat(sprintf("Unique cases: %d\n", n_distinct(runs$case_id)))
cat(sprintf("LLMs evaluated: %d\n", n_distinct(runs$llm_display)))

cat("\n--- Overall Performance ---\n")
cat(sprintf("Mean defensibility score: %.3f\n", mean(runs$score)))
cat(sprintf("Defensible rate (score=1): %.1f%%\n", mean(runs$score == 1) * 100))
cat(sprintf("Partial rate (score=0.5): %.1f%%\n", mean(runs$score == 0.5) * 100))
cat(sprintf("Liability rate (score=0): %.1f%%\n", mean(runs$score == 0) * 100))

cat("\n--- Cost Statistics ---\n")
cat(sprintf("Mean cost per recommendation: $%.2f\n", mean(runs$total_cost, na.rm = TRUE)))
cat(sprintf("Mean procedure count: %.1f\n", mean(runs$procedure_count, na.rm = TRUE)))
cat(sprintf("Cost-score correlation (individual): r = %.3f, %s\n",
            cor_cost$estimate, format_p(cor_cost$p.value)))
cat(sprintf("Cost-score correlation (model-level): r = %.2f, %s\n",
            cor_cost_model$estimate, format_p(cor_cost_model$p.value)))

cat("\n--- Top Model (GPT-5.2) ---\n")
top <- llm_summary %>% filter(llm_display == "GPT-5.2")
cat(sprintf("Mean score: %.3f\n", top$mean_score))
cat(sprintf("Defensible rate: %.1f%%\n", top$defensible_rate * 100))
cat(sprintf("Mean cost: $%.2f\n", top$mean_cost))
cat(sprintf("CPDR: $%.2f\n", top$cpdr))

cat("\n--- Bottom Model (GPT-4o) ---\n")
bottom <- llm_summary %>% filter(llm_display == "GPT-4o")
cat(sprintf("Mean score: %.3f\n", bottom$mean_score))
cat(sprintf("Defensible rate: %.1f%%\n", bottom$defensible_rate * 100))
cat(sprintf("Mean cost: $%.2f\n", bottom$mean_cost))

cat("\n--- Model Rankings by Mean Score ---\n")
for (i in 1:nrow(llm_summary)) {
  row <- llm_summary[i, ]
  cat(sprintf("%d. %s: %.2f (def=%.0f%%, cost=$%.0f)\n",
              i, row$llm_display, row$mean_score, row$defensible_rate * 100, row$mean_cost))
}

# ============================================
# SUPPLEMENTARY FIGURE: Procedure Categories Heatmap
# ============================================

# Load procedure category data from Python analysis
proc_avg <- read.csv("../results/procedure_avg_counts.csv")
proc_summary <- read.csv("../results/procedure_per_case_summary.csv")

# Define model order by defensibility (high to low)
model_order <- c("GPT-5.2", "GPT-5.2-Concise", "Grok 4.1", "Claude Sonnet 4.5",
                 "Claude Sonnet 4", "DeepSeek R1", "Gemini 3 Pro",
                 "Mistral Large", "Gemini 2.0 Flash", "Qwen3 30B",
                 "Qwen 2.5 72B", "Claude 3.5 Haiku", "Llama 4 Scout",
                 "Llama 4 Maverick", "GPT-4o", "GPT-4o Mini")

# Reshape for heatmap
proc_long <- proc_avg %>%
  select(model_display, lab, imaging, consultation, procedure) %>%
  pivot_longer(cols = c(lab, imaging, consultation, procedure),
               names_to = "category", values_to = "avg_count") %>%
  mutate(
    model_display = factor(model_display, levels = rev(model_order)),
    category = factor(category, levels = c("lab", "imaging", "consultation", "procedure"),
                     labels = c("Labs", "Imaging", "Consultations", "Procedures"))
  )

# Create heatmap
fig_supp_procedures <- ggplot(proc_long, aes(x = category, y = model_display, fill = avg_count)) +
  geom_tile(color = "white", linewidth = 1) +
  geom_text(aes(label = sprintf("%.1f", avg_count)), color = "black", size = 4) +
  scale_fill_gradient(low = "white", high = "#2166ac", name = "Avg per case") +
  labs(
    title = "Average Procedures Ordered Per Case by Model and Category",
    subtitle = "Higher values indicate more procedures ordered in each category",
    x = "Procedure Category",
    y = "Model"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(color = "gray40", size = 10),
    panel.grid = element_blank(),
    axis.text.x = element_text(angle = 0, hjust = 0.5)
  )

# Add total bar chart as panel B
proc_summary_plot <- proc_summary %>%
  mutate(model_display = factor(model_display, levels = rev(model_order)))

fig_supp_proc_total <- ggplot(proc_summary_plot, aes(x = model_display, y = as.numeric(mean_total_procs))) +
  geom_col(fill = "#2166ac", width = 0.7) +
  geom_text(aes(label = sprintf("%.1f", as.numeric(mean_total_procs))), hjust = -0.1, size = 3.5) +
  coord_flip() +
  labs(
    title = "Total Procedures Per Case",
    x = NULL,
    y = "Mean procedures"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    axis.text.y = element_blank()
  ) +
  xlim(rev(levels(proc_summary_plot$model_display))) +
  ylim(0, 14)

# Combine panels
fig_supp_procedures_combined <- fig_supp_procedures + fig_supp_proc_total +
  plot_layout(widths = c(3, 1))

ggsave("output/fig_supp_procedures.pdf", fig_supp_procedures_combined, width = 10, height = 6)

cat("\n=== Procedure Analysis ===\n")
cat("Mean procedures per case by model:\n")
print(proc_summary %>% select(model_display, mean_total_procs))

cat("\n" %+% paste(rep("=", 60), collapse = "") %+% "\n")
cat("STATISTICAL TESTS FOR PAPER CLAIMS\n")
cat(paste(rep("=", 60), collapse = "") %+% "\n")

cat("\n--- Overall Model Effect ---\n")
cat(sprintf("Kruskal-Wallis: chi-sq = %.2f, df = %d, %s\n",
            kw_test$statistic, kw_test$parameter, format_p(kw_test$p.value)))

cat("\n--- Key Pairwise Comparisons (Bonferroni-adjusted) ---\n")
# GPT-5.2 vs each other model
gpt5_comparisons <- pairwise_results %>% filter(model1 == "GPT-5.2")
for (i in 1:nrow(gpt5_comparisons)) {
  row <- gpt5_comparisons[i, ]
  sig <- ifelse(row$p_adjusted < 0.05, " *", "")
  cat(sprintf("GPT-5.2 vs %s: diff = %.2f, %s%s\n",
              row$model2, row$diff, format_p(row$p_adjusted), sig))
}

cat("\n--- Response Length ---\n")
cat(sprintf("GPT-5.2 mean: %.0f chars (SD = %.0f)\n", mean(gpt5_length), sd(gpt5_length)))
cat(sprintf("GPT-4o mean: %.0f chars (SD = %.0f)\n", mean(gpt4o_length), sd(gpt4o_length)))
cat(sprintf("t-test: t = %.2f, %s\n", length_ttest$statistic, format_p(length_ttest$p.value)))
cat(sprintf("Cohen's d = %.2f\n", cohens_d_length))

cat("\n--- Readability ---\n")
cat(sprintf("GPT-5.2 vs Grok 4.1: %s\n", format_p(read_ttest$p.value)))

cat("\n--- Correlations with P-values (Individual-level) ---\n")
cat(sprintf("Length-Defensibility: r = %.3f [%.3f, %.3f], %s\n",
            cor_length$estimate, cor_length$conf.int[1], cor_length$conf.int[2], format_p(cor_length$p.value)))
cat(sprintf("Readability-Defensibility: r = %.3f [%.3f, %.3f], %s\n",
            cor_readability$estimate, cor_readability$conf.int[1], cor_readability$conf.int[2], format_p(cor_readability$p.value)))
cat(sprintf("Cost-Defensibility: r = %.3f [%.3f, %.3f], %s\n",
            cor_cost$estimate, cor_cost$conf.int[1], cor_cost$conf.int[2], format_p(cor_cost$p.value)))
cat(sprintf("Reasoning-Defensibility: r = %.3f [%.3f, %.3f], %s\n",
            cor_reasoning$estimate, cor_reasoning$conf.int[1], cor_reasoning$conf.int[2], format_p(cor_reasoning$p.value)))

cat("\n--- Correlations (Model-level, n=7) ---\n")
cat(sprintf("Length-Defensibility: r = %.2f, %s\n", cor_length_model$estimate, format_p(cor_length_model$p.value)))
cat(sprintf("Readability-Defensibility: r = %.2f, %s\n", cor_read_model$estimate, format_p(cor_read_model$p.value)))
cat(sprintf("Cost-Defensibility: r = %.2f, %s\n", cor_cost_model$estimate, format_p(cor_cost_model$p.value)))
cat(sprintf("Reasoning-Defensibility: r = %.2f, %s\n", cor_reason_model$estimate, format_p(cor_reason_model$p.value)))

cat("\n--- Jurisdiction Effect ---\n")
cat(sprintf("Kruskal-Wallis: chi-sq = %.2f, %s\n", jurisdiction_kw$statistic, format_p(jurisdiction_kw$p.value)))
cat(sprintf("UK vs US: %s\n", format_p(uk_us$p.value)))
cat(sprintf("UK vs NZ: %s\n", format_p(uk_nz$p.value)))
cat(sprintf("US vs NZ: %s\n", format_p(us_nz$p.value)))

cat("\n--- Malpractice Category Effect ---\n")
cat(sprintf("Kruskal-Wallis: chi-sq = %.2f, %s\n", mal_kw$statistic, format_p(mal_kw$p.value)))

cat("\n--- Cost Comparisons ---\n")
cat(sprintf("GPT-5.2 vs GPT-4o: %s\n", format_p(cost_test$p.value)))
cat(sprintf("GPT-5.2 vs Gemini 3 Pro: %s\n", format_p(cost_test_gemini$p.value)))
cat(sprintf("Claude Sonnet 4 vs Gemini 3 Pro: %s\n", format_p(cost_test_claude$p.value)))

if (exists("cohens_kappa") && exists("kappa_ci")) {
  cat("\n--- Test-Retest Reliability ---\n")
  cat(sprintf("Cohen's kappa = %.2f, 95%% CI [%.2f, %.2f]\n", cohens_kappa, kappa_ci[1], kappa_ci[2]))
  cat(sprintf("Exact agreement = %.1f%%\n", exact_agreement * 100))
}

sink()

# ============================================
# SUPPLEMENTARY FIGURE: Case Flow Diagram (CONSORT-style)
# ============================================

library(grid)

# Create flow diagram data
flow_data <- data.frame(
  stage = c("Identified", "Extracted", "Pre-exclusion", "Excluded", "Analyzed"),
  n_cases = c(NA, 276, 276, 60, 216),
  n_runs = c(NA, 3868, 3868, NA, 2663)
)

# Create the flow diagram using ggplot
create_flow_diagram <- function() {
  # Box positions (x, y coordinates)
  boxes <- data.frame(
    id = 1:6,
    label = c(
      "Cases identified from\nlegal databases\n(BAILII, CourtListener, NZLII)",
      "Cases extracted with\nLLM pipeline\nn = 276 cases",
      "Case-model evaluation runs\nn = 3,868 runs\n(13 LLMs × 276 cases)",
      "Cases excluded:\nAll models scored 0 or NA\nn = 60 cases (22%)",
      "Analyzed benchmark\nn = 216 cases\nn = 2,663 valid runs",
      "Additional duplicate runs\nfor test-retest reliability\nn = 264 case-model pairs"
    ),
    x = c(0.5, 0.5, 0.5, 0.8, 0.5, 0.8),
    y = c(0.9, 0.72, 0.54, 0.45, 0.27, 0.18),
    width = c(0.4, 0.35, 0.35, 0.3, 0.35, 0.3),
    height = c(0.12, 0.1, 0.1, 0.1, 0.1, 0.1),
    fill = c("#e8f4f8", "#e8f4f8", "#e8f4f8", "#ffeaea", "#d4edda", "#fff3cd")
  )

  # Arrow connections
  arrows <- data.frame(
    x_start = c(0.5, 0.5, 0.5, 0.5),
    y_start = c(0.84, 0.67, 0.49, 0.49),
    x_end = c(0.5, 0.5, 0.5, 0.65),
    y_end = c(0.77, 0.59, 0.32, 0.45)
  )

  p <- ggplot() +
    # Draw boxes
    geom_rect(data = boxes,
              aes(xmin = x - width/2, xmax = x + width/2,
                  ymin = y - height/2, ymax = y + height/2,
                  fill = fill),
              color = "gray40", linewidth = 0.5) +
    # Add text
    geom_text(data = boxes,
              aes(x = x, y = y, label = label),
              size = 3, lineheight = 0.9) +
    # Draw arrows
    geom_segment(data = arrows,
                 aes(x = x_start, y = y_start, xend = x_end, yend = y_end),
                 arrow = arrow(length = unit(0.15, "cm"), type = "closed"),
                 linewidth = 0.5) +
    scale_fill_identity() +
    coord_fixed(ratio = 1, xlim = c(0, 1), ylim = c(0.1, 0.96)) +
    theme_void() +
    theme(plot.margin = margin(10, 10, 10, 10))

  return(p)
}

flow_fig <- create_flow_diagram()
ggsave("output/fig_supp_flow.pdf", flow_fig, width = 8, height = 10)
cat("Saved: output/fig_supp_flow.pdf\n")

# ============================================
# SUPPLEMENTARY FIGURE: Response Verbosity Distribution by LLM
# ============================================

# Categorize responses by character-count verbosity bins
runs_verbosity <- runs %>%
  filter(!is.na(recommendation_length)) %>%
  mutate(
    verbosity_category = case_when(
      recommendation_length < 1000 ~ "Concise (<1000)",
      recommendation_length < 2500 ~ "Moderate (1000-2500)",
      recommendation_length < 5000 ~ "Detailed (2500-5000)",
      recommendation_length < 10000 ~ "Verbose (5000-10000)",
      TRUE ~ "Extremely Verbose (>10000)"
    ),
    verbosity_category = factor(verbosity_category,
      levels = c("Concise (<1000)", "Moderate (1000-2500)",
                 "Detailed (2500-5000)", "Verbose (5000-10000)",
                 "Extremely Verbose (>10000)"))
  )

# Calculate per-model percentages
llm_verbosity <- runs_verbosity %>%
  count(llm_display, verbosity_category) %>%
  group_by(llm_display) %>%
  mutate(pct = n / sum(n)) %>%
  ungroup()

# Order models by proportion of concise+moderate responses
concise_order <- llm_verbosity %>%
  filter(verbosity_category %in% c("Concise (<1000)", "Moderate (1000-2500)")) %>%
  group_by(llm_display) %>%
  summarise(concise_pct = sum(pct), .groups = "drop") %>%
  arrange(concise_pct) %>%
  pull(llm_display)

llm_verbosity <- llm_verbosity %>%
  mutate(llm_display = factor(llm_display, levels = concise_order))

fig_supp_verbosity <- ggplot(llm_verbosity, aes(x = llm_display, y = pct, fill = verbosity_category)) +
  geom_col(position = "stack", alpha = 0.9, width = 0.7) +
  coord_flip() +
  scale_fill_manual(
    values = c("Concise (<1000)" = "#2ecc71",
               "Moderate (1000-2500)" = "#27ae60",
               "Detailed (2500-5000)" = "#f39c12",
               "Verbose (5000-10000)" = "#e67e22",
               "Extremely Verbose (>10000)" = "#e74c3c"),
    name = "Verbosity"
  ) +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(
    title = "Response Verbosity Distribution by LLM",
    subtitle = "Green = concise, Red = excessively verbose",
    x = NULL,
    y = "Percentage of Responses"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(color = "gray40", size = 10),
    legend.position = "right"
  )

ggsave("output/fig_supp_verbosity.pdf", fig_supp_verbosity, width = 10, height = 6)
ggsave("output/fig_supp_verbosity.png", fig_supp_verbosity, width = 10, height = 6, dpi = 300)
cat("Saved: output/fig_supp_verbosity.pdf\n")
