###################################
# Spine March 2024
###################################

library(lme4)
require(lmerTest)
library(car)
library(pbkrtest)
library(MuMIn)
library(sjPlot)
library(ggplot2)
library(modelr)
library(gridExtra)
library(cowplot)

# mixed effects model without interactions

model1.bmd <- lmer(BMD ~ Gender + Age + BMI + Level
                   + (1 | ScanID), data = spine.dat)
summary(model1.bmd)
confint(model1.bmd)

# validation
rand.eff <- ranef(model1.bmd)[["ScanID"]][["(Intercept)"]]
qqPlot(rand.eff, dist = "norm", pch = 19,
         mean = mean(rand.eff), sd = sd(rand.eff),
         xlab = "Theoretical quantiles", ylab = "Empirical quantiles",
         main = "Q-Q plot of random intercept")

qqPlot(resid(model1.bmd))

# rescaled outcome -> no visible effect
spine.dat$BMD.cent <- scale(spine.dat$BMD)

#------------

# full model with 2-way interactions


modelfull.bmd <- lmer(BMD ~ Gender * Age
                            + Gender * BMI
                            + Gender * Level
                            + Age * BMI
                            + Age * Level
                            + BMI * Level
                            + (1 | ScanID), data = spine.dat)
summary(modelfull.bmd)

plot(modelfull.bmd)
qqPlot(resid(modelfull.bmd))


# model selection backwards

step(modelfull.bmd, alpha.random = 0.05, alpha.fixed = 0.01)


model.final.bmd <- lmer(BMD ~ Gender * Age
                        # + Gender * BMI
                        + Gender * Level
                        + Age * BMI
                        + Age * Level
                        + BMI * Level
                        + (1 | ScanID), data = spine.dat)

summary(model.final.bmd)


KRmodcomp(modelfull.bmd, model.final.bmd)


plot(model.final.bmd)
qqPlot(resid(model.final.bmd))



# diagnostics
pdf("Analyses/Plots/model_residuals.pdf")
plot(model.final.bmd, Level ~ resid(.), abline = 0 ) 
dev.off()

plot(model.final.bmd, resid(., type = "pearson") ~ fitted(.) | Level, id = 0.05, 
         adj = -0.3, pch = 20, col = "gray40")

r.squaredGLMM(model.final.bmd)

pdf("Analyses/Plots/interaction_Age_Gender.pdf")
plot_model(model.final.bmd, type = "pred", terms = c("Age","Gender"))
dev.off()

pdf("Analyses/Plots/interaction_Age_BMI.pdf")
plot_model(model.final.bmd, type = "pred", terms = c("Age","BMI"))
dev.off()

# on pre-defined BMI-levels with additional parameter Gender
summary(spine.dat$BMI)
pdf("Analyses/Plots/interaction_Age_BMI_Gender.pdf")
plot_model(model.final.bmd, type = "pred", terms = c("Age","BMI [15,25,35]", "Gender"))
dev.off()


# Interaction plots per Level

pdf("Analyses/Plots/interaction_Age_Gender_Level.pdf")
plot_model(model.final.bmd, type = "pred", terms = c("Age","Gender","Level"))
dev.off()

pdf("Analyses/Plots/interaction_Age_BMI_Level.pdf")
plot_model(model.final.bmd, type = "pred", terms = c("Age","BMI [15,25,35]","Level"))
dev.off()




#----------------------------------------
# using quadratic function for age
#----------------------------------------

# full model with 2-way interactions
modelfull.bmd2 <- lmer(BMD ~ Gender * Age.trans
                      + Gender * BMI
                      + Gender * Level
                      + Age.trans * BMI
                      + Age.trans * Level
                      + BMI * Level
                      + (1 | ScanID), data = spine.dat)
summary(modelfull.bmd2)
plot(modelfull.bmd2)
qqPlot(resid(modelfull.bmd2))


# model selection backwards
step(modelfull.bmd2, alpha.random = 0.05, alpha.fixed = 0.01)


model.final.bmd2 <- lmer(BMD ~ Gender * Age.trans
                        # + Gender * BMI
                        + Gender * Level
                        + Age.trans * BMI
                        + Age.trans * Level
                        + BMI * Level
                        + (1 | ScanID), data = spine.dat)

summary(model.final.bmd2)


KRmodcomp(modelfull.bmd2, model.final.bmd2)

plot(model.final.bmd2)
qqPlot(resid(model.final.bmd2))



# diagnostics
pdf("Analyses/Plots/model_residuals_Age_transformed.pdf")
plot(model.final.bmd2, Level ~ resid(.), abline = 0 ) 
dev.off()

plot(model.final.bmd2, resid(., type = "pearson") ~ fitted(.) | Level, id = 0.05, 
     adj = -0.3, pch = 20, col = "gray40")

r.squaredGLMM(model.final.bmd2)


# interaction plots -> need for re-transformation


# Backtransform
# Age.re <- 18 +/- 6*sqrt(9-Age.trans)
spine.dat$Age.re <- 18+6*sqrt(9-spine.dat$Age.trans)

trans <- function(x){
  x - x^2/36
}

back.trans <- function(x){
  18+6*sqrt(9-x)
}

# interaction between age & gender

out <- plot_model(model.final.bmd2, type = "pred", terms = c("Age.trans [-166:9]","Gender"))

# take confidence bounds out of plot_model
fem.x <- back.trans(out$data$x[out$data$group == "female"])
mal.x <- back.trans(out$data$x[out$data$group == "male"])
fem.y <- out$data$predicted[out$data$group == "female"]
mal.y <- out$data$predicted[out$data$group == "male"]
fem.l <- out$data$conf.low[out$data$group == "female"]
fem.u <- out$data$conf.high[out$data$group == "female"]
mal.l <- out$data$conf.low[out$data$group == "male"]
mal.u <- out$data$conf.high[out$data$group == "male"]

# generate new data for prediction on original age scale                            
new_dat <- data.frame(Age = c(fem.x, mal.x),
                      Gender = c(rep("female",length(fem.x)),rep("male",length(mal.x))),
                      BMD = c(fem.y, mal.y),
                      BMD.low = c(fem.l,mal.l),
                      BMD.up = c(fem.u,mal.u))

pdf("Analyses/Plots/interaction_Age_Gender_Age_transformed.pdf")
ggplot(new_dat, aes(x = Age, y = BMD, color = Gender)) +
  geom_line()  +
  geom_ribbon(aes(ymax = BMD.up, ymin = BMD.low, fill = Gender), alpha = 0.3, colour = NA) +
  geom_line(aes(color = Gender)) 
dev.off()


# interaction between age and bmi

out <- plot_model(model.final.bmd2, type = "pred", terms = c("Age.trans [-166:9]","BMI [15,25,35]"))

# take confidence bounds out of plot_model
high.x <- back.trans(out$data$x[out$data$group == "35"])
med.x <- back.trans(out$data$x[out$data$group == "25"])
low.x <- back.trans(out$data$x[out$data$group == "15"])

high.y <- out$data$predicted[out$data$group == "35"]
med.y <- out$data$predicted[out$data$group == "25"]
low.y <- out$data$predicted[out$data$group == "15"]

high.l <- out$data$conf.low[out$data$group == "35"]
high.u <- out$data$conf.high[out$data$group == "35"]
med.l <- out$data$conf.low[out$data$group == "25"]
med.u <- out$data$conf.high[out$data$group == "25"]
low.l <- out$data$conf.low[out$data$group == "15"]
low.u <- out$data$conf.high[out$data$group == "15"]

# generate new data for prediction on original age scale                            
new_dat <- data.frame(Age = c(high.x, med.x, low.x),
                      BMI = c(rep("35",length(high.x)),
                              rep("25",length(med.x)),
                              rep("15", length(low.x))),
                      BMD = c(high.y, med.y, low.y),
                      BMD.low = c(high.l,med.l,low.l),
                      BMD.up = c(high.u,med.u,low.u))

pdf("Analyses/Plots/interaction_Age_BMI_Age_transformed.pdf")
ggplot(new_dat, aes(x = Age, y = BMD, color = BMI)) +
  geom_line()  +
  geom_ribbon(aes(ymax = BMD.up, ymin = BMD.low, fill = BMI), alpha = 0.3, colour = NA) +
  geom_line(aes(color = BMI)) 
dev.off()


# interaction between age, bmi and level

out <- plot_model(model.final.bmd2, type = "pred",
                   terms = c("Age.trans [-166:9]","BMI [15,25,35]","Level"))

# loop over different levels
lev <- levels(out$data$facet)

plot_list <- list()


for (i in 1:length(lev))
{
  # take confidence bounds out of plot_model
  high.x <- back.trans(out$data$x[out$data$group == "35" & out$data$facet == lev[i]])
  med.x <- back.trans(out$data$x[out$data$group == "25" & out$data$facet == lev[i]])
  low.x <- back.trans(out$data$x[out$data$group == "15" & out$data$facet == lev[i]])
  
  high.y <- out$data$predicted[out$data$group == "35" & out$data$facet == lev[i]]
  med.y <- out$data$predicted[out$data$group == "25" & out$data$facet == lev[i]]
  low.y <- out$data$predicted[out$data$group == "15" & out$data$facet == lev[i]]
  
  high.l <- out$data$conf.low[out$data$group == "35" & out$data$facet == lev[i]]
  high.u <- out$data$conf.high[out$data$group == "35" & out$data$facet == lev[i]]
  med.l <- out$data$conf.low[out$data$group == "25" & out$data$facet == lev[i]]
  med.u <- out$data$conf.high[out$data$group == "25" & out$data$facet == lev[i]]
  low.l <- out$data$conf.low[out$data$group == "15" & out$data$facet == lev[i]]
  low.u <- out$data$conf.high[out$data$group == "15" & out$data$facet == lev[i]]
  
  # generate new data for prediction on original age scale                            
  new_dat <- data.frame(Age = c(high.x, med.x, low.x),
                        BMI = c(rep("35",length(high.x)),
                                rep("25",length(med.x)),
                                rep("15", length(low.x))),
                        BMD = c(high.y, med.y, low.y),
                        BMD.low = c(high.l,med.l,low.l),
                        BMD.up = c(high.u,med.u,low.u))
  
  p <- ggplot(new_dat, aes(x = Age, y = BMD, color = BMI)) +
        geom_line()  +
        geom_ribbon(aes(ymax = BMD.up, ymin = BMD.low, fill = BMI), alpha = 0.3, colour = NA) +
        geom_line(aes(color = BMI)) +
        ggtitle(lev[i])

    plot_list[[i]] <- p
  
}

# function to extract the legend from a ggplot object
get_legend <- function(my_plot) {
  tmp <- ggplot_gtable(ggplot_build(my_plot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)
}

# extract legend from one of the plots
legend <- get_legend(plot_list[[1]])

# remove legend from all plots
plot_list <- lapply(plot_list, function(x) x + theme(legend.position = "none"))

# arrange plots in a grid with 5 columns and 4 rows
combined_plots <- arrangeGrob(
  grobs = plot_list, 
  ncol = 5,
  nrow = 4
)

pdf("Analyses/Plots/interaction_Age_BMI_Level_Age_transformed.pdf",
    width = 14, height = 10)
# add global legend
grid.arrange(
  combined_plots, legend, 
  ncol = 2, 
  widths = c(10, 2) # adjust the width ratio as needed
)
dev.off()



# interaction between age, gender and level

out <- plot_model(model.final.bmd2, type = "pred",
                  terms = c("Age.trans [-166:9]","Gender","Level"))

# loop over different levels
lev <- levels(out$data$facet)

plot_list <- list()


for (i in 1:length(lev))
{
  # take confidence bounds out of plot_model
  fem.x <- back.trans(out$data$x[out$data$group == "female" & out$data$facet == lev[i]])
  mal.x <- back.trans(out$data$x[out$data$group == "male" & out$data$facet == lev[i]])
  fem.y <- out$data$predicted[out$data$group == "female" & out$data$facet == lev[i]]
  mal.y <- out$data$predicted[out$data$group == "male" & out$data$facet == lev[i]]
  fem.l <- out$data$conf.low[out$data$group == "female" & out$data$facet == lev[i]]
  fem.u <- out$data$conf.high[out$data$group == "female" & out$data$facet == lev[i]]
  mal.l <- out$data$conf.low[out$data$group == "male" & out$data$facet == lev[i]]
  mal.u <- out$data$conf.high[out$data$group == "male" & out$data$facet == lev[i]]
  
  # generate new data for prediction on original age scale                            
  new_dat <- data.frame(Age = c(fem.x, mal.x),
                        Gender = c(rep("female",length(fem.x)),rep("male",length(mal.x))),
                        BMD = c(fem.y, mal.y),
                        BMD.low = c(fem.l,mal.l),
                        BMD.up = c(fem.u,mal.u))


  p <- ggplot(new_dat, aes(x = Age, y = BMD, color = Gender)) +
    geom_line()  +
    geom_ribbon(aes(ymax = BMD.up, ymin = BMD.low, fill = Gender), alpha = 0.3, colour = NA) +
    geom_line(aes(color = Gender)) +
    ggtitle(lev[i])
  
  plot_list[[i]] <- p
  
}

# extract legend from one of the plots
legend <- get_legend(plot_list[[1]])

# remove legend from all plots
plot_list <- lapply(plot_list, function(x) x + theme(legend.position = "none"))

# arrange plots in a grid with 5 columns and 4 rows
combined_plots <- arrangeGrob(
  grobs = plot_list, 
  ncol = 5,
  nrow = 4
)

pdf("Analyses/Plots/interaction_Age_Gender_Level_Age_transformed.pdf",
    width = 14, height = 10)
# add global legend
grid.arrange(
  combined_plots, legend, 
  ncol = 2, 
  widths = c(10, 2) # adjust the width ratio as needed
)
dev.off()
