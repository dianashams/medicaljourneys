##############################################################################
## ELSA -> age-grid long format for discrete-time multi-outcome survival    ##
##############################################################################
#
# Input : wide data frame `d`, one row per person
#           yob, yod              birth / death year (decimal)
#           date1..dateW          interview dates (decimal years), NA if missed
#           <condition>           AGE at first diagnosis, NA if never
#           sex, edu, wealth, bmi_latest, child_*
#
# Output: long table, one row per (person, age-bin) while under observation
#           at_risk_<k>  1 if still at risk for k in this bin
#           event_<k>    1 if k first diagnosed in this bin
#           prev_<k>     1 if k was already present at the START of this bin
#
# Time scale is AGE. Entry = age at first interview (left truncation).
# Exit  = age at last interview, or age at death.
#
# Duplicate long/short families:
#   long  (angina, heartattack, ...)  = self-reported AGE AT DIAGNOSIS
#   short (angin, hrtatt, ...)        = age first REPORTED in an ELSA wave
# The short one lags the long one by 1-6y, differentially by condition
# (parkinson 1.1y, arthritis 5.8y). We take the EARLIEST available across all
# columns mapping to a group. Conditions with no long-name partner -- hibp,
# hchol, lung, asthma, catract, osteo, hip, chf -- carry the wave lag
# unmitigated, so treat directed edges involving those with caution.

library(dplyr)
library(tidyr)

# #  ELSA_short_df.csv - start from here
# f2 <- "C:/Users/dinab/Desktop/PhD Projects/Ensemble methods/GitHub_App/medicaljourneys/ELSA data/ELSA_short_df.csv"
# d2<- read.csv(f2)
# names(d2)[1:63]
# apply(d2, 2, FUN = function(x) sum(is.na(x)))
# 
# # add BMI from the harmonized file
# d3 <- haven::read_dta("C:/Users/dinab/Desktop/PostDoc/ELSA files/UKDA-5050-stata/stata/stata13_se/gh_elsa_h.dta")
# names(d3)
# 
# bmicols <- c("r2mbmi", "r4mbmi", "r6mbmi", "r8mbmi")
# d2 <- d2 %>%
#   left_join(
#     d3 %>% select(idauniq, all_of(bmicols)),
#     by = "idauniq"
#   )
# 
# sum(d3$rachshlt %in% c(1,2,3,4,5,6)) / dim(d3)[1]
# 
# d2 <- d2 %>% mutate(bmi_latest = coalesce(r8mbmi, r6mbmi, r4mbmi, r2mbmi))
# names(d2)[c(1:62, 104)]

# check first and last wave participated
# iwstat_cols <- paste0("r", 1:10, "iwstat")
# d3 <- d3 %>%
#   mutate(
#     across(all_of(iwstat_cols), as.numeric)
#   ) %>%
#   mutate(
#     first_wave = apply(
#       select(., all_of(iwstat_cols)),
#       1,
#       function(x) {
#         waves <- which(!is.na(x) & x == 1)
#         if (length(waves) == 0) NA_integer_ else min(waves)
#       }
#     ),
#     latest_wave = apply(
#       select(., all_of(iwstat_cols)),
#       1,
#       function(x) {
#         waves <- which(!is.na(x) & x == 1)
#         if (length(waves) == 0) NA_integer_ else max(waves)
#       }
#     )
#   )
# 
# d2 <- d2 %>%
#   left_join(
#     d3 %>% select(idauniq, all_of(iwstat_cols)),
#     by = "idauniq"
#   )

# write.csv(d2, f2)

f2 <- "C:/Users/dinab/Desktop/PhD Projects/Ensemble methods/GitHub_App/medicaljourneys/ELSA data/ELSA_short_df.csv"
elsa_data <- read.csv(f2)


# ----------------------------------------------------------------- settings

AGE_MIN <- 50      # first age bin starts here
AGE_MAX <- 90      # last bin ends here
BIN     <- 2       # bin width in years (matches biennial waves)

MIN_EVENTS <- 200  # groups below this are reported as too thin to model

# ------------------------------------------------------------- the event map
# column in `d`  ->  event group.  "!remove" / NA are dropped.

remap <- c(
  # --- cardiovascular ---
  "angina"      = "cvd",
  "heartattack" = "cvd",
  "chf"         = "cvd",
  "heart"       = "cvd",
  "angin"       = "cvd",
  "hrtatt"      = "cvd",
  "hrtrhm"      = "!remove",     # arrhythmia: finding, not disease
  "hrtmr"       = "!remove",     # murmur: often benign / incidental
  "cvd"         = "!remove",     # derived from the above -> would double-count
  "stroke"      = "stroke",
  "strok"       = "stroke",
  # --- metabolic ---
  "hibp"        = "hypertension",
  "hchol"       = "!remove",
  "diabetes"    = "diabetes",
  "diab"        = "diabetes",
  # --- respiratory ---
  "lung"        = "asthma_respir",
  "asthma"      = "asthma_respir",
  # --- cancer ---
  "cancer"      = "cancer",
  "cancr"       = "cancer",
  # --- musculoskeletal ---
  "arthritis"   = "arthritis",
  "arthr"       = "arthritis",
  "osteo"       = "osteoporosis",
  "hip"         = "hip_replacement",
  # --- mental / neuro ---
  "psych"       = "mental_health",
  "depression"  = "mental_health",
  "parkinson"   = "neuro",
  "parkin"      = "neuro",
  "alzheimer"   = "neuro",
  "dementia"    = "neuro",
  # --- other ---
  "catract"     = "cataract",
  # --- terminal ---
  "died"        = "died",
  # --- dropped ---
  "hyster"      = "!remove",
  "last_period" = "!remove",
  "born"        = "!remove",
  "remove!"     = "!remove"
)

# childhood conditions: CONTEXT ONLY, never events.
# Mapping these to adult event channels would place an event at age ~6, which
# the pipeline classifies as prevalent at entry -- silently removing those
# people from the adult risk set for that condition for life. Childhood asthma
# often resolves; adult COPD is a different disease.
CHILD_MAP <- c(
  "child_asthma"   = "child_asthma_respir",
  "child_resp"     = "child_asthma_respir",
  "child_epilepsy" = "child_epilepsy",
  "child_psych"    = "child_psych",
  "child_bones"    = "child_bones",
  "child_diab"     = "child_diabetes",
  "child_heart"    = "child_heart",
  "child_leuk"     = "child_cancer",
  "child_infect"   = "!remove",
  "child_allerg"   = "!remove",
  "child_hdache"   = "!remove",
  "child_appdcts"  = "!remove"
)

STATIC <- c("sex", "edu", "wealth", "bmi_latest")

# ---------------------------------------------------------------- helpers

pmin_na <- function(m) {
  out <- suppressWarnings(apply(m, 1, min, na.rm = TRUE))
  out[is.infinite(out)] <- NA_real_
  out
}

# which column supplied the earliest age (for the reporting-lag sensitivity)
src_of <- function(m, cols) {
  apply(m, 1, function(r)
    if (all(is.na(r))) NA_character_
    else cols[which.min(replace(r, is.na(r), Inf))])
}

# ------------------------------------------- 1. collapse columns into groups

remap <- remap[!is.na(remap) & remap != "!remove"]
remap <- remap[names(remap) %in% names(elsa_data)]           # ignore absent columns

groups <- sort(unique(unname(remap)))
groups <- c(setdiff(groups, "died"), "died")         # death last, it absorbs

for (g in groups) {
  cols <- names(remap)[remap == g]
  m    <- as.matrix(elsa_data[, cols, drop = FALSE])
  elsa_data[[paste0("age_", g)]] <- pmin_na(m)
  if (length(cols) > 1) elsa_data[[paste0("src_", g)]] <- src_of(m, cols)
}

# death from yod if no `died` age column resolved
if (!"age_died" %in% names(elsa_data) || all(is.na(elsa_data$age_died)))
  elsa_data$age_died <- ifelse(is.na(elsa_data$yod), NA_real_, elsa_data$yod - elsa_data$yob)

EV      <- groups
age_col <- setNames(paste0("age_", EV), EV)

message("event groups (", length(EV), "): ", paste(EV, collapse = ", "))

# ---------------------------------------------------------- 2. child context

CHILD_MAP <- CHILD_MAP[CHILD_MAP != "!remove"]
CHILD_MAP <- CHILD_MAP[names(CHILD_MAP) %in% names(elsa_data)]
child_groups <- sort(unique(unname(CHILD_MAP)))

for (g in child_groups) {
  cols <- names(CHILD_MAP)[CHILD_MAP == g]
  elsa_data[[g]] <- as.integer(rowSums(!is.na(elsa_data[, cols, drop = FALSE])) > 0)
}
elsa_data$child_n <- rowSums(elsa_data[, child_groups, drop = FALSE])

# ------------------------------------------------------ 3. entry / exit ages

date_cols <- grep("^date[0-9]+$", names(elsa_data), value = TRUE)
date_cols <- date_cols[order(as.integer(sub("date", "", date_cols)))]
stopifnot(length(date_cols) > 0)

dates      <- as.matrix(elsa_data[, date_cols])
first_date <- suppressWarnings(apply(dates, 1, function(r)
{ r <- r[!is.na(r)]; if (length(r)) min(r) else NA_real_ }))
last_date  <- suppressWarnings(apply(dates, 1, function(r)
{ r <- r[!is.na(r)]; if (length(r)) max(r) else NA_real_ }))

elsa_data  <- elsa_data %>%
  mutate(entry_age = first_date - yob,
         obs_end   = last_date  - yob,
         exit_age  = pmin(obs_end, ifelse(is.na(age_died), Inf, age_died))) %>%
  filter(!is.na(entry_age), !is.na(exit_age), exit_age > entry_age)

# ------------------------------------------------------------ 4. age binning

n_bins <- as.integer((AGE_MAX - AGE_MIN) / BIN)
bin_of <- function(a) {
  b <- floor((a - AGE_MIN) / BIN) + 1L
  ifelse(is.na(a) | b < 1L | b > n_bins, NA_integer_, b)
}

elsa_data  <- elsa_data %>%
  mutate(entry_bin = bin_of(pmax(entry_age, AGE_MIN)),
         exit_bin  = bin_of(pmin(exit_age, AGE_MAX - 1e-6))) %>%
  filter(!is.na(entry_bin), !is.na(exit_bin), exit_bin >= entry_bin)

message(sprintf("n = %d people | %d age bins (%g-%g, width %g) | entry age %.0f-%.0f",
                nrow(elsa_data), n_bins, AGE_MIN, AGE_MAX, BIN,
                min(elsa_data$entry_age), max(elsa_data$entry_age)))

# ------------------------------------------------------ 5. expand to the grid

long <- elsa_data %>%
  select(idauniq, entry_bin, exit_bin) %>%
  rowwise() %>%
  reframe(idauniq = idauniq, bin = seq(entry_bin, exit_bin)) %>%
  left_join(elsa_data, by = "idauniq") %>%
  mutate(age_mid = AGE_MIN + (bin - 0.5) * BIN)

# ------------------------------------------- 6. event / at-risk / prev flags

for (k in EV) {
  a  <- long[[age_col[[k]]]]
  kb <- bin_of(a)                                    # bin of diagnosis
  prevalent <- !is.na(a) & (a <= long$entry_age)     # present before entry
  
  long[[paste0("event_",   k)]] <- as.integer(!is.na(kb) & kb == long$bin & !prevalent)
  long[[paste0("prev_",    k)]] <- as.integer(prevalent | (!is.na(kb) & long$bin > kb))
  long[[paste0("at_risk_", k)]] <- as.integer(!prevalent & (is.na(kb) | long$bin <= kb))
}

# death absorbs: once dead, nothing else is at risk
dead_now <- long$prev_died == 1L
for (k in setdiff(EV, "died")) long[[paste0("at_risk_", k)]][dead_now] <- 0L

# ------------------------------------------------------------ 7. covariates

long <- long %>% mutate(age_c = (age_mid - 70) / 10)

keep <- c("idauniq", "bin", "age_mid", "age_c", "entry_bin", "exit_bin",
          STATIC, child_groups, "child_n",
          grep("^src_", names(long), value = TRUE),
          paste0("event_", EV), paste0("prev_", EV), paste0("at_risk_", EV))

long <- long %>% select(any_of(keep)) %>% arrange(idauniq, bin)

# ------------------------------------------------------------------ 8. checks

if ("bmi_latest" %in% STATIC)
  warning("bmi_latest is measured at the LAST nurse visit, so for most people ",
          "it POSTDATES their events. Used as a baseline covariate this is a ",
          "look-ahead leak and will inflate discrimination. Replace with a ",
          "wave-1 BMI, or merge longitudinal nurse-visit BMI, before reporting.")

cat("\n--- incident events per group (after left truncation) ---\n")
ev_counts <- long %>%
  summarise(across(starts_with("event_"), sum)) %>%
  pivot_longer(everything(), names_to = "event", values_to = "n") %>%
  mutate(event = sub("event_", "", event)) %>%
  arrange(n)
print(ev_counts, n = Inf)

thin <- ev_counts %>% filter(n < MIN_EVENTS)
if (nrow(thin))
  message("\nTOO THIN to support a per-age-bin hazard (< ", MIN_EVENTS,
          " events): ", paste(thin$event, collapse = ", "),
          "\n  -> merge these, or drop them, before modelling.")

cat("\n--- prevalent at entry (%) ---\n")
print(long %>% group_by(idauniq) %>% slice_min(bin, n = 1) %>% ungroup() %>%
        summarise(across(starts_with("prev_"), ~ round(100 * mean(.x), 1))) %>%
        pivot_longer(everything(), names_to = "event", values_to = "pct_at_entry") %>%
        mutate(event = sub("prev_", "", event)) %>%
        arrange(desc(pct_at_entry)), n = Inf)

cat("\n--- person-bins per age bin (thin bins -> noisy baseline hazard) ---\n")
print(long %>% count(bin, age_mid), n = n_bins)

cat("\n--- which column supplied the earliest age ---\n")
for (s in grep("^src_", names(long), value = TRUE)) {
  tb <- table(long %>% group_by(idauniq) %>% slice_min(bin, n = 1) %>%
                ungroup() %>% pull(!!s))
  cat(sprintf("%-28s %s\n", s, paste(sprintf("%s:%d", names(tb), tb), collapse = "  ")))
}

cat("\n--- digit heaping: diagnosis ages mod 5 (flat = none; spike at 0 = rounding) ---\n")
for (k in EV) {
  a <- elsa_data[[age_col[[k]]]]
  if (sum(!is.na(a)) < 50) next
  h <- table(floor(a) %% 5)
  cat(sprintf("%-20s %s\n", k,
              paste(sprintf("%s:%s", names(h), round(100 * h / sum(h))), collapse = "  ")))
}

# sanity: an event must never fire outside the risk set
for (k in EV) {
  bad <- sum(long[[paste0("event_", k)]] == 1 & long[[paste0("at_risk_", k)]] == 0)
  if (bad) warning(sprintf("%s: %d events outside the risk set", k, bad))
}

long <- long %>%
  mutate(
    wealth = replace(wealth, is.na(wealth), mean(wealth, na.rm = TRUE)),
    edu    = replace(edu,    is.na(edu),    mean(edu, na.rm = TRUE)),
    bmi_latest    = replace(bmi_latest,    is.na(bmi_latest),    mean(bmi_latest, na.rm = TRUE))
  )

write.csv(long, "C:/Users/dinab/Desktop/PhD Projects/Ensemble methods/GitHub_App/medicaljourneys/ELSA data/elsa_long_agegrid.csv", row.names = FALSE)
message("\nwritten: elsa_long_agegrid.csv  (", nrow(long), " person-bins)")



