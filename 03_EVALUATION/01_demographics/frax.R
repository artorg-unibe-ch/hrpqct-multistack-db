library(ggplot2)
library(dplyr)
library(showtext)
library(hrbrthemes)

# Add Google fonts
font_add_google("Outfit", "title_font")
font_add_google("Cabin", "body_font")
showtext_auto()

title_font <- "title_font"
body_font <- "body_font"

# Function to calculate summary statistics
calculate_stats <- function(df, column) {
  stats <- list(
    mean = mean(df[[column]], na.rm = TRUE),
    sd = sd(df[[column]], na.rm = TRUE),
    median = median(df[[column]], na.rm = TRUE),
    lq = quantile(df[[column]], 0.25, na.rm = TRUE),
    uq = quantile(df[[column]], 0.75, na.rm = TRUE)
  )
  return(stats)
}

# Function to perform p-value calculation
perform_p_value <- function(df1, df2, column, type = "mean") {
  if (type == "mean") {
    # Perform Student's t-test for continuous variables presented with mean (sd)
    test <- t.test(df1[[column]], df2[[column]], na.rm = TRUE)
  } else if (type == "median") {
    # Perform Wilcoxon-Mann-Whitney test for continuous variables presented with median [lq, uq]
    test <- wilcox.test(df1[[column]], df2[[column]], na.rm = TRUE)
  }
  return(test$p.value)
}


create_latex_table <- function(df_female, df_male, stats_major_f, stats_major_m, stats_hip_f, stats_hip_m, p_value_major, p_value_hip, label) {
  format_p_value <- function(p) {
    if (p < 0.001) {
      return("p<0.001")
    } else if (p < 0.01) {
      return(sprintf("%.3f", p))
    } else {
      return(sprintf("%.2f", p))
    }
  }
  
  latex_table <- sprintf("
\\begin{table}[h!]
\\centering
\\scriptsize
\\begin{tabular}{lccc}
\\toprule
 & Females \\((n=%d)\\) & Males \\((n=%d)\\) & p-value* \\\\
\\midrule
FRAX: major osteoporotic fracture (percent}) & & & \\\\
\\quad Mean (sd) & %.2f (%.2f) & %.2f (%.2f) & %s \\\\
\\quad Median [lq, uq] & %.2f [%.2f, %.2f] & %.2f [%.2f, %.2f] & %s \\\\
FRAX: Hip Fracture (%%) & & & \\\\
\\quad Mean (sd) & %.2f (%.2f) & %.2f (%.2f) & %s \\\\
\\quad Median [lq, uq] & %.2f [%.2f, %.2f] & %.2f [%.2f, %.2f] & %s \\\\
\\bottomrule
\\end{tabular}
\\caption{FRAX ten year probability of fracture (percent) for major osteoporotic fractures and hip fractures, statistics for %s. *Student's t test for continuous variables presented with mean (sd) and the Wilcoxon-Mann-Whitney test for continuous variables presented with median [lq, uq]. \\textit{sd, standard deviation; lq, lower quartile; uq, upper quartile}.}
\\label{tab:%s}
\\end{table}
", 
  nrow(df_female), nrow(df_male), 
  stats_major_f$mean, stats_major_f$sd, stats_major_m$mean, stats_major_m$sd, format_p_value(p_value_major$mean),
  stats_major_f$median, stats_major_f$lq, stats_major_f$uq, stats_major_m$median, stats_major_m$lq, stats_major_m$uq, format_p_value(p_value_major$median),
  stats_hip_f$mean, stats_hip_f$sd, stats_hip_m$mean, stats_hip_m$sd, format_p_value(p_value_hip$mean),
  stats_hip_f$median, stats_hip_f$lq, stats_hip_f$uq, stats_hip_m$median, stats_hip_m$lq, stats_hip_m$uq, format_p_value(p_value_hip$median),
  label, label)

  cat(latex_table)
}

# Read the CSV file
df_young <- read.csv("03_EVALUATION/01_demographics/frax_young.csv")
df_old <- read.csv("03_EVALUATION/01_demographics/frax_old_new.csv")

# Filter data by sex
df_young_male <- df_young %>% filter(sex == "male")
df_young_female <- df_young %>% filter(sex == "female")

df_old_male <- df_old %>% filter(sex == "male")
df_old_female <- df_old %>% filter(sex == "female")

# Calculate statistics for the 2 groups
stats_frax_major_young_m <- calculate_stats(df_young_male, "major.osteoporotic.risk")
stats_frax_major_young_f <- calculate_stats(df_young_female, "major.osteoporotic.risk")

stats_frax_major_old_m <- calculate_stats(df_old_male, "major.osteoporotic.risk")
stats_frax_major_old_f <- calculate_stats(df_old_female, "major.osteoporotic.risk")

stats_frax_hip_young_m <- calculate_stats(df_young_male, "hip.fracture.risk")
stats_frax_hip_young_f <- calculate_stats(df_young_female, "hip.fracture.risk")

stats_frax_hip_old_m <- calculate_stats(df_old_male, "hip.fracture.risk")
stats_frax_hip_old_f <- calculate_stats(df_old_female, "hip.fracture.risk")

# Perform p-value calculation
p_value_major_young <- list(
  mean = perform_p_value(df_young_male, df_young_female, "major.osteoporotic.risk", "mean"),
  median = perform_p_value(df_young_male, df_young_female, "major.osteoporotic.risk", "median")
)

p_value_hip_young <- list(
  mean = perform_p_value(df_young_male, df_young_female, "hip.fracture.risk", "mean"),
  median = perform_p_value(df_young_male, df_young_female, "hip.fracture.risk", "median")
)

p_value_major_old <- list(
  mean = perform_p_value(df_old_male, df_old_female, "major.osteoporotic.risk", "mean"),
  median = perform_p_value(df_old_male, df_old_female, "major.osteoporotic.risk", "median")
)

p_value_hip_old <- list(
  mean = perform_p_value(df_old_male, df_old_female, "hip.fracture.risk", "mean"),
  median = perform_p_value(df_old_male, df_old_female, "hip.fracture.risk", "median")
)


# create LaTeX table for young group
create_latex_table(
    df_young_female, df_young_male,
    stats_frax_major_young_f, stats_frax_major_young_m, stats_frax_hip_young_f, stats_frax_hip_young_m,
    p_value_major_young, p_value_hip_young, "under 40")

# create LaTeX table for old group
create_latex_table(
    df_old_female, df_old_male,
    stats_frax_major_old_f, stats_frax_major_old_m, stats_frax_hip_old_f, stats_frax_hip_old_m,
    p_value_major_old, p_value_hip_old, "over 40"
)