library(ggplot2)
library(dplyr)
library(showtext)
library(hrbrthemes)
library(stats)

# Add Google fonts
font_add_google("Outfit", "title_font")
font_add_google("Cabin", "body_font")
showtext_auto()

title_font <- "title_font"
body_font <- "body_font"

# Create the directory if it doesn't exist
if (!dir.exists("demographics/r_plots")) {
  dir.create("demographics/r_plots")
}

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

create_latex_table <- function(df_female, df_male, stats_bmd_female, stats_bmd_male, stats_bmi_female, stats_bmi_male, stats_age_female, stats_age_male, p_value_bmd, p_value_bmi, p_value_age, label) {
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
Age (years) & & & \\\\
\\quad Mean (sd) & %.2f (%.2f) & %.2f (%.2f) & %s \\\\
\\quad Median [lq, uq] & %.2f [%.2f, %.2f] & %.2f [%.2f, %.2f] & %s \\\\
BMI (kg/m²) & & & \\\\
\\quad Mean (sd) & %.2f (%.2f) & %.2f (%.2f) & %s \\\\
\\quad Median [lq, uq] & %.2f [%.2f, %.2f] & %.2f [%.2f, %.2f] & %s \\\\
Femoral neck aBMD (g/cm²) & & & \\\\
\\quad Mean (sd) & %.2f (%.2f) & %.2f (%.2f) & %s \\\\
\\quad Median [lq, uq] & %.2f [%.2f, %.2f] & %.2f [%.2f, %.2f] & %s \\\\
\\bottomrule
\\end{tabular}
\\caption{Age (years), Femoral neck aBMD (g/cm²), and BMI (kg/m²) statistics for %s. *Student's t test for continuous variables presented with mean (sd) and the Wilcoxon-Mann-Whitney test for continuous variables presented with median [lq, uq]. \\textit{sd, standard deviation; lq, lower quartile; uq, upper quartile; aBMD, areal bone mineral density}.}
\\label{tab:%s}
\\end{table}
", 
  nrow(df_female), nrow(df_male), 
  stats_age_female$mean, stats_age_female$sd, stats_age_male$mean, stats_age_male$sd, format_p_value(p_value_age$mean),
  stats_age_female$median, stats_age_female$lq, stats_age_female$uq, stats_age_male$median, stats_age_male$lq, stats_age_male$uq, format_p_value(p_value_age$median),
  stats_bmi_female$mean, stats_bmi_female$sd, stats_bmi_male$mean, stats_bmi_male$sd, format_p_value(p_value_bmi$mean),
  stats_bmi_female$median, stats_bmi_female$lq, stats_bmi_female$uq, stats_bmi_male$median, stats_bmi_male$lq, stats_bmi_male$uq, format_p_value(p_value_bmi$median),
  stats_bmd_female$mean, stats_bmd_female$sd, stats_bmd_male$mean, stats_bmd_male$sd, format_p_value(p_value_bmd$mean),
  stats_bmd_female$median, stats_bmd_female$lq, stats_bmd_female$uq, stats_bmd_male$median, stats_bmd_male$lq, stats_bmd_male$uq, format_p_value(p_value_bmd$median),
  label, label)

  cat(latex_table)
}

# Read the CSV file
# df <- read.csv("00_DB/HR-pQCT_database_2024-09-12.csv")
df <- read.csv("00_DB/HR-pQCT_database_expanded_2025-06-16_16-18.csv")
# Calculate BMI
df$BMI <- df$Weight / ((df$Height / 100)^2)

# Stratify for age
df_under_40 <- df %>% filter(Age < 40)
df_over_40 <- df %>% filter(Age >= 40)

# Stratify for gender
df_under_40_male <- df_under_40 %>% filter(Gender == "Male")
# print ages of df_under_40_male
df_under_40_female <- df_under_40 %>% filter(Gender == "Female")
df_over_40_male <- df_over_40 %>% filter(Gender == "Male")
df_over_40_female <- df_over_40 %>% filter(Gender == "Female")

# Calculate statistics for under 40
stats_bmd_under_40_male <- calculate_stats(df_under_40_male, "Femoral.neck.BMD..g.cm2.")
stats_bmd_under_40_female <- calculate_stats(df_under_40_female, "Femoral.neck.BMD..g.cm2.")
stats_bmi_under_40_male <- calculate_stats(df_under_40_male, "BMI")
stats_bmi_under_40_female <- calculate_stats(df_under_40_female, "BMI")
stats_age_under_40_male <- calculate_stats(df_under_40_male, "Age")
stats_age_under_40_female <- calculate_stats(df_under_40_female, "Age")

# Calculate statistics for over 40
stats_bmd_over_40_male <- calculate_stats(df_over_40_male, "Femoral.neck.BMD..g.cm2.")
stats_bmd_over_40_female <- calculate_stats(df_over_40_female, "Femoral.neck.BMD..g.cm2.")
stats_bmi_over_40_male <- calculate_stats(df_over_40_male, "BMI")
stats_bmi_over_40_female <- calculate_stats(df_over_40_female, "BMI")
stats_age_over_40_male <- calculate_stats(df_over_40_male, "Age")
stats_age_over_40_female <- calculate_stats(df_over_40_female, "Age")

# Perform p-value calculations
p_value_bmd_under_40 <- list(
  mean = perform_p_value(df_under_40_male, df_under_40_female, "Femoral.neck.BMD..g.cm2.", "mean"),
  median = perform_p_value(df_under_40_male, df_under_40_female, "Femoral.neck.BMD..g.cm2.", "median")
)

p_value_bmi_under_40 <- list(
  mean = perform_p_value(df_under_40_male, df_under_40_female, "BMI", "mean"),
  median = perform_p_value(df_under_40_male, df_under_40_female, "BMI", "median")
)

p_value_age_under_40 <- list(
  mean = perform_p_value(df_under_40_male, df_under_40_female, "Age", "mean"),
  median = perform_p_value(df_under_40_male, df_under_40_female, "Age", "median")
)

p_value_bmd_over_40 <- list(
  mean = perform_p_value(df_over_40_male, df_over_40_female, "Femoral.neck.BMD..g.cm2.", "mean"),
  median = perform_p_value(df_over_40_male, df_over_40_female, "Femoral.neck.BMD..g.cm2.", "median")
)

p_value_bmi_over_40 <- list(
  mean = perform_p_value(df_over_40_male, df_over_40_female, "BMI", "mean"),
  median = perform_p_value(df_over_40_male, df_over_40_female, "BMI", "median")
)

p_value_age_over_40 <- list(
  mean = perform_p_value(df_over_40_male, df_over_40_female, "Age", "mean"),
  median = perform_p_value(df_over_40_male, df_over_40_female, "Age", "median")
)

# Create LaTeX tables for under 40
create_latex_table(df_under_40_female, df_under_40_male, stats_bmd_under_40_female, stats_bmd_under_40_male, stats_bmi_under_40_female, stats_bmi_under_40_male, stats_age_under_40_female, stats_age_under_40_male, p_value_bmd_under_40, p_value_bmi_under_40, p_value_age_under_40, "under 40")

# Create LaTeX tables for over 40
create_latex_table(df_over_40_female, df_over_40_male, stats_bmd_over_40_female, stats_bmd_over_40_male, stats_bmi_over_40_female, stats_bmi_over_40_male, stats_age_over_40_female, stats_age_over_40_male, p_value_bmd_over_40, p_value_bmi_over_40, p_value_age_over_40, "over 40")
# Add subgroup labels
df_under_40$Subgroup <- "Under 40"
df_over_40$Subgroup <- "Over 40"

# Combine all dataframes
df_combined <- bind_rows(df_under_40, df_over_40)

# Create histograms for each column in each subgroup
for (column in names(df_combined)) {
  if (is.numeric(df_combined[[column]])) {
    # Filter out non-finite values
    df_combined_filtered <- df_combined %>% filter(is.finite(!!sym(column)))
    
    p <- df_combined_filtered %>%
      ggplot(aes_string(x = column, fill = "Subgroup")) +
      geom_histogram(color = "white", alpha = 0.6, position = 'stack') +
      scale_fill_manual(values = c("Under 40" = "#69b3a2", "Over 40" = "#404080")) +
      theme_minimal() +
      theme(
        panel.background = element_rect(fill = "white", color = "white"),
        axis.title = element_text(family = body_font, size = 24),
        axis.text = element_text(family = body_font, size = 24),
        legend.position = "top",
        legend.title = element_blank(),
        legend.spacing = unit(0.5, 'cm'),
        legend.key.height = unit(0.5, 'cm'),
        legend.key.width = unit(0.7, 'cm'),
        legend.text = element_text(family = body_font, size = 20, face = 'plain', color = "grey10"),
        plot.title = element_text(family = title_font, size = 24, face = "bold", margin = margin(20, 0, 10, 0)),
        plot.subtitle = element_text(family = body_font, size = 18, color = "grey15", margin = margin(10, 0, 20, 0)),
        plot.caption = element_text(family = body_font, size = 20, color = "grey40", hjust = 0.5, margin = margin(20, 0, 0, 0)),
        plot.background = element_rect(color = "white", fill = "white"),
        plot.margin = margin(10, 20, 10, 20)
      ) +
      labs(
        fill = "",
        title = paste("Histogram of", column, "by Age Group"),
        x = column,
        y = "Frequency"
      )
    
    # Save the plot
    ggsave(filename = paste0("03_EVALUATION/01_demographics/r_plots/histogram_", column, ".png"), plot = p, width = 5, height = 5)
  }
}

showtext_auto(FALSE)