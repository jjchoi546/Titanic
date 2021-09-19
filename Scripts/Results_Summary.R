#Comparison of the misclassification rates between the models
models <- c("Base", "Pruned Tree")
mse <- c(cv.base_mse, cv.pTree_mse)
sae <- c(cv.base_sae, cv.pTree_sae)

#Results location
results_name_csv <- "Results_Comparison.csv"

dir_out <- "output/Predictions Review/"

png(file.path(dir_out, "MSE Result Comparison of Models.png"))
n <- length(models)
plot(1:n, mse, xlab="Models", ylab="Predicion MSE Rates", type="b", xaxt="n")
points(which.min(mse), mse[which.min(mse)], col="red", cex=2, pch=20)
axis(1, at=1:n, labels=models, las=1)
dev.off()

png(file.path(dir_out, "SAE Result Comparison of Models.png"))
n <- length(models)
plot(1:n, sae, xlab="Models", ylab="Predicion SAE Rates", type="b", xaxt="n")
points(which.min(sae), sae[which.min(sae)], col="red", cex=2, pch=20)
axis(1, at=1:n, labels=models, las=1)
dev.off()
