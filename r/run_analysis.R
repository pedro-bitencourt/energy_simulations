# Import libraries
library(dplyr)
library(mgcv)
library(lubridate)
library(ggplot2)

if(interactive()) {
  experiment_name <- "zero_hydro"
} else{
  args <- commandArgs(trailingOnly = TRUE)
  experiment_name <- args[1]
}

simulation_folder <- paste0("/Users/pedrobitencourt/Projects/energy_simulations/sim/", experiment_name)

# Define paths
raw_folder <- paste0(simulation_folder, "/raw/")
graphics_folder <- paste0("/Users/pedrobitencourt/Projects/energy_simulations/figures/",experiment_name,"/runs/")

# Read investment results
solver_results <- read.csv(paste0(simulation_folder, "/results/solver_results.csv"))



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
  
  # Step 1 and a half. Parse if needed
  if(!"production_salto" %in% names(data)) {
    data$production_salto <- 0
    hydro_present <- FALSE
  }else{
    hydro_present <- TRUE
  }
  if(!"demand" %in% names(data)){
    data$demand <- data$production_demand
  }
  if(!"water_level_salto" %in% names(data)){
    data$water_level_salto <- 0
    message("water level variable not found, inputting 0")
  }
  
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
  data$positive_lost_load <- as.integer(data$demand > data$production_total + 0.1)
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
  plot_density(data, "net_demand", bw = 1)
  plot_density(data, "production_salto", bw = 1)
  plot_density(data %>% filter(production_thermal > 5), "production_thermal", bw = 1, overwrite = TRUE)
  plot_density(data, "production_wind", bw = 1)
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
  if(hydro_present) {
    npr_probability_lost_load()
  }
  
  # Step 4. Compute some statistics
  message("Computing statistics")
 
  # Load capacities for the run 
  capacities <- solver_results %>% filter(name == run_name)
  
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
  
  # Compute means
  # Production
  results$mean_thermal_production <- mean(data$production_thermal)
  results$mean_wind_production <- mean(data$production_wind)
  results$mean_solar_production <- mean(data$production_solar)
  results$mean_salto_production <- mean(data$production_salto)
  results$mean_lost_load <- mean(data$lost_load)
  results$mean_excess <- mean(data$production_excedentes)
  
  # Price
  results$mean_price <- mean(data$marginal_cost)
  
  # Standard deviations
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
  
  ## Quartiles
  # Price distribution
  results$q25_prob_price_4000 <- quantile(avg_per_scenario$prob_price_4000, 0.25, names=FALSE) * 8760
  results$q50_prob_price_4000 <- quantile(avg_per_scenario$prob_price_4000, 0.50, names=FALSE) * 8760
  results$q75_prob_price_4000 <- quantile(avg_per_scenario$prob_price_4000, 0.75, names=FALSE) * 8760
  results$q25_prob_price_193_4000 <- quantile(avg_per_scenario$prob_price_193_4000, 0.25, names=FALSE) * 8760
  results$q50_prob_price_193_4000 <- quantile(avg_per_scenario$prob_price_193_4000, 0.50, names=FALSE) * 8760
  results$q75_prob_price_193_4000 <- quantile(avg_per_scenario$prob_price_193_4000, 0.75, names=FALSE) * 8760
  
  # LCOE: E[Total cost]/E[demand]
  wind_fixed_costs_mw_hour <- 8.22
  solar_fixed_costs_mw_hour <- 4.62
  thermal_fixed_costs_mw_hour <- 6.93
  
  results$lcoe <- (capacities$thermal_capacity * thermal_fixed_costs_mw_hour +
    capacities$wind_capacity * wind_fixed_costs_mw_hour +
    capacities$solar_capacity * solar_fixed_costs_mw_hour +
    results$mean_variable_cost) / mean(data$demand)
  
  # Capture rates: E[pq]/E[q]
  results$thermal_capture_rate <- mean(data$marginal_cost * data$production_thermal)/mean(data$production_thermal)
  results$wind_capture_rate <- mean(data$marginal_cost * data$production_wind)/mean(data$production_wind)
  results$solar_capture_rate <- mean(data$marginal_cost * data$production_solar)/mean(data$production_solar)
  
  # Avg capture rate: E[p]/E[q]
  results$avg_capture_rate <- mean(data$marginal_cost * data$demand)/mean(data$demand)
  
  message("Finished ", run_name, " summary statistics:", results)
  return(results)
  }

run_files <- list.files(raw_folder, full.names = TRUE)

all_results <- list()
for(run_file in run_files) {
  run_results <- analyze_run(run_file)
  all_results <- rbind(all_results, run_results)
  print(all_results)
}

# Save to a CSV file
output_file <- file.path(simulation_folder, "results_summary.csv")
write.csv(all_results, output_file)
