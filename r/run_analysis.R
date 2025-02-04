gc()

rm(list = ls())
# Import libraries
library(dplyr)
library(mgcv)
library(lubridate)
library(ggplot2)

if(interactive()) {
  experiment_name <- "qturbmax"
  experiment_name <- "factor_compartir"
  simulation_folder <- paste0("/Users/pedrobitencourt/Projects/energy_simulations/simulations/", experiment_name)
} else{
  args <- commandArgs(trailingOnly = TRUE)
  simulation_folder <- args[1]
}


# Define paths
raw_folder <- paste0(simulation_folder, "/raw/")
graphics_folder <- paste0(simulation_folder, "/figures/runs/")


# Read investment results
investment_results <- read.csv(paste0(simulation_folder, "/investment_results.csv"))



analyze_run <- function(run_file) {
  # Step 0. Initialize run name and folders
  # Extract the name of the run
  run_name <- tools::file_path_sans_ext(basename(run_file))
  if(run_name == "salto_capacity_fc_0" | run_name == "salto_capacity_fc_5"){
    return(NULL)
  }
  message("starting ", run_name)
  # Set output folder for the run
  run_output_folder <- paste0(graphics_folder, run_name)
  dir.create(run_output_folder, recursive = TRUE)
  
  # Functions
  # plot_density: Estimates kernel densities, plots and saves it
  plot_density <- function(data, name, x_from = 0, x_to = NA,
                           bw = 1, overwrite = FALSE) {
    # Create title for the plot
    title <- paste("Density Plot", gsub("_", " ", tools::toTitleCase(name)))

        # Create output path
    output_path <- file.path(run_output_folder, paste0(name, ".png"))
    # Check if output exists; if yes, return
    if (file.exists(output_path) & overwrite == FALSE) {
      message("output_path already exists", output_path)
      return("skipping")
    }
    plot <- ggplot(data, aes(x = .data[[name]])) +
      geom_density(kernel = "epanechnikov", bw = bw, adjust = 1) +
      xlim(x_from, x_to) +
      ggtitle(title) +
      xlab(gsub("_", " ", tools::toTitleCase(name))) +
      ylab("Density")
    ggsave(output_path, plot)
    print(plot)
  }
 
  # Do a non-parametric regression on the probability of lost load
  npr_probability_lost_load <- function() {
    output_path <- file.path(run_output_folder, "predicted_probability_lost_load.png")
    # Check if output exists; if yes, return
    if (file.exists(output_path)) {
      message("output_path already exists", output_path)
      return("skipping")
    }
    model <- gam(positive_lost_load ~ s(water_level_salto), data = data, family = binomial)
    # Predict probabilities
    data$predicted <- predict(model, type = "response")
    # Generate a sequence of water_level_salto values for prediction
    x_range <- seq(min(data$water_level_salto), max(data$water_level_salto), length.out = 100)
    
    # Predict probabilities for the range of x values
    predicted_probs <- predict(model, newdata = data.frame(water_level_salto = x_range), type = "response")
    
    # Plot the predicted probabilities
    plot(x_range, predicted_probs, type = "l", main = "Predicted Probabilities from GAM",
         xlab = "Water Level (Salto)", ylab = "Predicted Probability", col = "blue", lwd = 2) 
    # Prepare the data for plotting
    plot_data <- data.frame(
 
           x = x_range, 
      predicted = predicted_probs
    )
    
    # Create the plot using ggplot2
    plot <- ggplot(plot_data, aes(x = x, y = predicted)) +
      geom_line(color = "blue", size = 1.2) +  # Add the predicted probabilities line
      ggtitle("Predicted Probability of Lost Load") +
      xlab("Water Level (Salto)") +
      ylab("Predicted Probability of Positive Lost Load") +
      theme_minimal() 
    
    # Save the plot to a file
    ggsave(output_path, plot = plot, width = 8, height = 6, dpi = 300)
  }
  # Step 1. Read file 
  data <- read.csv(run_file)
  
  # Step 2. Data manipulation
  # Create variables
  data$production_total <- data$production_wind +
                           data$production_solar +
                           data$production_salto +
                           data$production_thermal
  data$demand_minus_production <- data$demand - data$production_total
  data$net_demand <- data$demand - data$production_solar - data$production_wind

  data$lost_load <- pmax(data$demand_minus_production, 0)
  data$excess_production <- pmax(-data$demand_minus_production, 0)
  data$excess_fraction <- data$excess_production/data$production_total
  
  # Booleans
  data$thermal_active <- as.integer(data$production_thermal > 0.1)
  data$hydro_active <- as.integer(data$production_salto > 0.1)
  data$hydro_marginal <- as.integer(data$hydro_active & !data$thermal_active)
  data$thermal_marginal <- as.integer(data$thermal_active & !data$hydro_active)
  data$renewables_marginal <- as.integer(!data$thermal_active & !data$hydro_active)
  data$price_4000 <- as.integer(data$marginal_cost >= 4000)
  data$positive_lost_load <- as.integer(data$production_demand > data$production_total + 0.1)
  data$dummy_excess <- as.integer(data$excess_production > 0.1)
  data$price_193_4000 <- as.integer(data$marginal_cost > 193.1 & data$marginal_cost < 4000)
  
  # Round numeric variables to nearest .1
  data <- data %>% mutate(across(where(is.numeric), ~ round(.x,1)))
  
  # Step 3. Plotting
  # 3.a Plot densities
  # 3.a.i) Prices
  # All
  plot_density(data, "marginal_cost")
  #
  ## Conditional on no lost_load
  plot_density(data %>% filter(marginal_cost < 4000) %>% rename(marginal_cost_no_lost_load = marginal_cost), "marginal_cost_no_lost_load")
  #
  ## Conditional on hydro_marginal
  plot_density(data %>% filter(hydro_marginal == 1) %>% rename(marginal_cost_positive_hydro = marginal_cost),
               "marginal_cost_positive_hydro")
  
  ## 3.a.ii) Production
  # Net Demand
  plot_density(data, "net_demand", bw = 1)
  # Hydro
  plot_density(data, "production_salto", bw = 1)
  # Thermal
  plot_density(data %>% filter(production_thermal > 5), "production_thermal", bw = 1, overwrite = TRUE)
  # Wind
  plot_density(data, "production_wind", bw = 1)
  # Solar
  plot_density(data %>% filter(production_solar > 0), "production_solar", bw = 1)
  
  # 3.a.iii) Water level
  # Unconditional
  plot_density(data, "water_level_salto", x_from = 29, bw = 0.1, overwrite = TRUE)
  
  # Value of water
  temp <- data %>% filter(data$thermal_active == 1 & data$hydro_active == 0) %>% mutate(marginal_cost_vw = marginal_cost)
  plot_density(temp, "marginal_cost_vw", bw = 1, x_from = 0, x_to = 300)
  
  temp <- data %>% filter(marginal_cost > 195 & marginal_cost < 3000) %>% mutate(marginal_cost_vw_2 = marginal_cost)
  plot_density(temp, "marginal_cost_vw_2", bw = 1)
  
  
  # Conditional on hydro production
  # 3. Probability of lost load
  npr_probability_lost_load()
  
  # Step 4. Compute some statistics
  message("Computing statistics")
 
  # Load capacities for the run 
  capacities <- investment_results %>% filter(name == run_name)
  
  results <- list()
  
  results$name <- run_name
  results$exogenous_variable <- as.numeric(sub(".*_", "", run_name))
  
  # Compute probabilities of miscelanous events
  results$prob_hydro_active <- mean(data$hydro_active) * 8760
  results$prob_thermal_active <- mean(data$thermal_active) * 8760
  results$prob_water_level_31 <- mean(data$water_level_salto < 31) * 8760
  results$prob_water_level_33 <- mean(data$water_level_salto < 33) * 8760
  results$prob_hydro_marginal <- mean(data$hydro_marginal) * 8760
  results$prob_thermal_marginal <- mean(data$thermal_marginal) * 8760
  results$prob_renewables_marginal <- mean(data$renewables_marginal) * 8760
  results$prob_price_4000 <- mean(data$price_4000) * 8760
  results$prob_price_193_4000 <- mean(data$price_193_4000) * 8760
  results$prob_lost_load_wl_31 <- mean(data %>% filter(water_level_salto < 31) %>% pull(positive_lost_load)) * 8760
  results$prob_lost_load_wl_32 <- mean(data %>% filter(water_level_salto < 32) %>% pull(positive_lost_load)) * 8760
  results$prob_lost_load <- mean(data$lost_load > 0) * 8760
  
  results$thermal_production_conditional_active <- mean(data %>% filter(thermal_active == 1) %>% pull(production_thermal))
  
  results$thermal_profit <- mean((data$marginal_cost - 193) * data$production_thermal) / capacities$thermal_capacity
  results$thermal_profit_positive_ll <- mean((data$marginal_cost - 193) * data$production_thermal * data$positive_lost_load) / capacities$thermal_capacity
  results$thermal_profit_price_4000 <- mean((data$marginal_cost - 193) * data$production_thermal * data$price_4000)/ capacities$thermal_capacity
  results$thermal_profit_price_193_4000 <- mean((data$marginal_cost - 193) * data$production_thermal * data$price_193_4000)/ capacities$thermal_capacity
  
  results$mean_thermal_production <- mean(data$production_thermal)
  
  results$std_thermal_profit <- sd((data$marginal_cost - 193) * data$production_thermal) / capacities$thermal_capacity
  results$std_thermal_profit_blackout <- sd((data$marginal_cost - 193) * data$production_thermal * data$positive_lost_load) / capacities$thermal_capacity
  results$std_thermal_profit_price_193_4000 <- sd((data$marginal_cost - 193) * data$production_thermal * data$price_193_4000) / capacities$thermal_capacity
  
  results$price_conditional_price_193_4000 <- mean(data %>% filter(price_193_4000 == 1) %>% pull(marginal_cost))
  results$lost_load_cond_ll_0 <- mean(data %>% filter(positive_lost_load == 1) %>% pull(lost_load))
  
  results$mean_lost_load <- mean(data$lost_load)
  results$mean_variable_cost <- mean(data$variable_cost_thermal)
  
  results$excess_wind <- mean(data$excess_fraction * data$production_wind, na.rm = TRUE)
  results$excess_solar <- mean(data$excess_fraction * data$production_solar, na.rm = TRUE)
  results$excess_salto <- mean(data$excess_fraction * data$production_salto, na.rm = TRUE)
  
  # Distributions over scenarios
  avg_per_scenario <- data %>% 
    group_by(scenario) %>% 
    summarise(prob_price_4000 = mean(price_4000),
              prob_price_193_4000 = mean(price_193_4000))
  
  results$q25_prob_price_4000 <- quantile(avg_per_scenario$prob_price_4000, 0.25) * 8760
  results$q50_prob_price_4000 <- quantile(avg_per_scenario$prob_price_4000, 0.50) * 8760
  results$q75_prob_price_4000 <- quantile(avg_per_scenario$prob_price_4000, 0.75) * 8760
  
  results$q25_prob_price_193_4000 <- quantile(avg_per_scenario$prob_price_193_4000, 0.25) * 8760
  results$q50_prob_price_193_4000 <- quantile(avg_per_scenario$prob_price_193_4000, 0.50) * 8760
  results$q75_prob_price_193_4000 <- quantile(avg_per_scenario$prob_price_193_4000, 0.75) * 8760
  
  # LCOE
  results$lcoe <- (
    capacities$thermal_capacity_mw * capacities$thermal_fixed_costs_mw_hour +
    capacities$wind_capacity_mw * capacities$wind_fixed_costs_mw_hour +
    capacities$solar_capacity_mw * capacities$solar_fixed_costs_mw_hour +
    capacities$thermal_variable_costs_hour) / mean(data$production_demand)
  
    
  message("Finished ", run_name, " summary statistics:")
  return(results)
}

run_files <- list.files(raw_folder, full.names = TRUE)
all_results <- list()
for(run_file in run_files) {
  run_results <- analyze_run(run_file)
  all_results <- append(all_results, list(run_results))
}
# Convert the list of results to a dataframe
results_df <- do.call(rbind, lapply(all_results, as.data.frame))  # Each run becomes a row
results_df <- data.frame(results_df, row.names = NULL)  # Ensure a clean dataframe

# Save to a CSV file
output_file <- file.path(simulation_folder, "results_summary.csv")
write.csv(results_df, output_file)
