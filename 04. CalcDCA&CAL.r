# Currently Python does not have a package to do survival decision curve plotting, so we had to use the R language for that operation.
# To ensure uniformity in the image format, we saved the plotted data using the R language and proceeded to plot the graph using Python.
rm(list = ls())
library(ggDCA)
library(PredictABEL)
library(reticulate)
library(readxl)
library(rms)
library(survival)
library(dplyr)
library(stringr)
library(PredictABEL)
library(reticulate)
library(readxl)
library(rms)
library(survival)
Result_Folder <- "/mnt/Projects/Radiomics-CT/99.Manuscript/Result/"
Out_Folder <- "/mnt/Projects/Radiomics-CT/99.Manuscript/Data/Plot/"
df_Inst1_Y <- read.csv("/mnt/Projects/Radiomics-CT/99.Manuscript/Data/Inst.1.Y.csv")
df_InstO_Y <- read.csv("/mnt/Projects/Radiomics-CT/99.Manuscript/Data/Inst.Other.Y.csv")
path_Stage <- "/mnt/Projects/Radiomics-CT/99.Manuscript/Data/Stages.csv"
df_stage <- read.csv(path_Stage)
df_stage <- df_stage[!duplicated(df_stage$PatientID), ]
df_stage$APASL <- NULL
cut_time = 721
boost_time = 1000
dat.1 <- merge(df_Inst1_Y, df_stage, by = "PatientID", all.x = TRUE)
dat.1 <- dat.1[match(df_Inst1_Y$PatientID, dat.1$PatientID), ]
dat.O <- merge(df_InstO_Y, df_stage, by = "PatientID", all.x = TRUE)
dat.O <- dat.O[match(df_InstO_Y$PatientID, dat.O$PatientID), ]
Eval <- function(text) {
    return(eval(parse(text = text)))
}
models <- list.dirs(Result_Folder, full.names = F)
for (m in models) {
    if (m != "" & m != "CV") {
        d.1 <- read.csv(paste0(Result_Folder, m, "/Inst.1.csv"), row.names = 1)
        x <- d.1[length(rownames(d.1)), ]
        Eval(paste0("dat.1$", m, " <<- t(x)"))
        d.O <- read.csv(paste0(Result_Folder, m, "/Inst.Other.csv"), row.names = 1)
        xo <- d.O[length(rownames(d.O)), ]
        Eval(paste0("dat.O$", m, " <<- t(xo)"))
    }
}
col_time <- "T"
col_event <- "E"
FullNA <- function(df) {
    handle_column <- function(column) {
        if (is.numeric(column)) {
            return(ifelse(is.na(column), mean(column, na.rm = TRUE), column))
        } else {
            column <- ifelse(column == "" | str_trim(column) == "", NA, column)
            return(ifelse(is.na(column), "Unknown", column))
        }
    }
    colnames <- colnames(df)
    df <- data.frame(lapply(df, handle_column), stringsAsFactors = FALSE)
    colnames(df) <- colnames
    return(df)
}
set.seed(666)
CalcDCA <- function(dat, fn) {
    dat <- dat[sample(nrow(dat)), ]
    dat$T <- as.numeric(dat$T)
    dat$E <- as.logical(dat$E)
    if (any(is.na(dat))) {
        dat <- FullNA(dat)
    }
    cnams <- colnames(dat)
    cnams <- cnams[!(cnams %in% c(col_time, col_event, "PatientID"))]
    dd <- datadist(dat)
    assign("dat", dat, envir = .GlobalEnv)
    assign("dd", dd, envir = .GlobalEnv)
    options(datadist = "dd")
    for (c in cnams) {
        i <- Eval(paste0("cph(Surv(", col_time, ", ", col_event, ") ~ ", c, ", data = dat)"))
        assign(c, i, envir = .GlobalEnv)
    }
    ddca <- Eval(paste0("dca(", paste0(cnams, collapse = ", ", sep = ""), ")"))
    ddca <- data.frame(ddca)
    file.remove(paste0(Out_Folder, fn))
    write.csv(ddca, file = paste0(Out_Folder, fn))
}
CalcDCA(dat.1, "DCA.Inst.1.csv")
CalcDCA(dat.O, "DCA.Inst.Other.csv")
d1 <- read.csv(paste0(Result_Folder, "/DIDANet/Inst.1.csv"), row.names = 1)
do <- read.csv(paste0(Result_Folder, "/DIDANet/Inst.Other.csv"), row.names = 1)
d1 <- d1[length(rownames(d1)), ]
do <- do[length(rownames(do)), ]
df_Inst1_Y$V = t(d1)
df_InstO_Y$V = t(do)
drawCalCurve <- function(dat, title) {
    dat$E[dat$T>=cut_time] = F
    dat$T[dat$T>=cut_time] = cut_time
    cph_TRAIN <- cph(Surv(T, E) ~ V, data = dat, surv = TRUE, x = TRUE, y = TRUE, time.inc = cut_time - 1)
    cal_1 <- calibrate(cph_TRAIN, u = cut_time - 1, cmethod = "KM", m = length(row.names(dat)) / 3, B = boost_time)
    return(cal_1)
}
re1 = drawCalCurve(df_Inst1_Y, title_name)
reO = drawCalCurve(df_InstO_Y, title_name)
reTe = drawCalCurve(df_Inst1_Y[1:40,], title_name)
reTr = drawCalCurve(df_Inst1_Y[40:length(rownames(df_Inst1_Y)) ,], title_name)
write.csv(re1, file="/mnt/Projects/Radiomics-CT/99.Manuscript/Data/Plot/CAL.Inst.1.csv")
write.csv(reO, file="/mnt/Projects/Radiomics-CT/99.Manuscript/Data/Plot/CAL.Inst.Other.csv")
write.csv(reTr, file="/mnt/Projects/Radiomics-CT/99.Manuscript/Data/Plot/CAL.Inst.1.Tr.csv")
write.csv(reTe, file="/mnt/Projects/Radiomics-CT/99.Manuscript/Data/Plot/CAL.Inst.1.Te.csv")