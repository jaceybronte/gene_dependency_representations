# Define colors and labels for the models
model_colors <- c(
  "pca" = "#e41a1c", 
  "ica" = "#377eb8", 
  "nmf" = "#4daf4a", 
  "vanillavae" = "#e4e716", 
  "betavae" = "#984ea3", 
  "betatcvae" = "#ff7f00"
)

model_labels <- c(
  "pca" = "PCA", 
  "ica" = "ICA", 
  "nmf" = "NMF", 
  "vanillavae" = "VAE",
  "betavae" = "βVAE",
  "betatcvae" = "βTCVAE"
)


# Custom theme function
custom_theme <- function() {
  theme(
    legend.position = "right",
    
    # --- Font sizes ---
    text = element_text(size = 30),
    axis.title = element_text(size = 34),
    axis.text = element_text(size = 28),
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
    legend.title = element_text(size = 30),
    legend.text = element_text(size = 28),
    plot.title = element_text(size = 40)
  )
}