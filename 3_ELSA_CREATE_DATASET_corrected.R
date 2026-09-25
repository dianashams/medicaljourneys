##############################################################################
## ELSA -> age-grid long format for discrete-time multi-outcome survival    ##
##############################################################################
#
# Input : wide data frame, one row per person
#           idauniq, yob, yod, died
#           date1..dateW          interview dates (decimal years)
#           r1iwstat..r10iwstat   harmonised response status per wave
#           r2mbmi, r4mbmi, ...   nurse-visit BMI (for the time-varying option)
#           <condition>           AGE at first diagnosis, NA if never
#           sex, edu, wealth, child_*
#
# Output: long table, one row per (person, age-bin) while under observation
#           at_risk_<k>  1 if still at risk for k in this bin
#           event_<k>    1 if k first diagnosed in this bin
#           prev_<k>     1 if k was already present at the START of this bin
#
# TIME SCALE IS AGE, so alpha[k, t] in the model is the age-specific baseline
# hazard -- interpretable, and comparable with published incidence curves.
#
# CENSORING comes from r{w}iwstat, NOT from max(date1..dateW). The date columns
# are populated for every wave regardless of attendance, so using them follows
# dropouts to the last wave as event-free: immortal person-time, and a
# downward-biased baseline hazard at older ages exactly where it matters.
#
# DEATH is resolved from several sources in order of quality; iwstat == 5
# recovers several times more deaths than the `died` / yod columns alone.
#
# DUPLICATE long/short families:
#   long  (angina, heartattack, ...)  = self-reported AGE AT DIAGNOSIS
#   short (angin, hrtatt, ...)        = age first REPORTED in an ELSA wave
# The short one lags the long one by 1-6y, DIFFERENTIALLY by condition
# (parkinson 1.1y, arthritis 5.8y). We take the earliest available across all
# columns mapping to a group. Conditions with no long-name partner -- hibp,
# lung, asthma, catract, osteo, hip, chf -- carry the wave lag unmitigated, so
# treat directed edges involving those with caution: differential lag can
# distort apparent ORDERING between conditions.

library(dplyr)
library(tidyr)

# ----------------------------------------------------------------- settings

PATH_IN  <- "C:/Users/dinab/Desktop/PhD Projects/Ensemble methods/GitHub_App/medicaljourneys/ELSA data/ELSA_short_df.csv"
PATH_OUT <- "C:/Users/dinab/Desktop/PhD Projects/Ensemble methods/GitHub_App/medicaljourneys/ELSA data/elsa_long_agegrid.csv"

AGE_MIN <- 50      # first age bin starts here
AGE_MAX <- 90      # last bin ends here
BIN     <- 2       # bin width in years (matches biennial waves)

MIN_EVENTS <- 200  # groups below this are flagged as too thin to model

RESP_CODES <- c(1) # harmonised iwstat: 1 = responded, alive.
# add 2 if proxy interviews should count as a response.
DEAD_CODE  <- 5    # 5 = died this wave  (6 = died at an earlier wave)

# BMI handling:
#   "timevarying" - nurse-visit BMI carried forward to each age bin  [BEST]
#   "first"       - earliest nurse BMI, as a fixed baseline covariate
#   "latest"      - latest nurse BMI  [LOOK-AHEAD LEAK: it postdates events]
BMI_MODE     <- "timevarying"
BMI_WAVES    <- c(2, 4, 6, 8)          # waves with a nurse visit
BMI_BACKFILL <- TRUE                   # bins before the first measurement get
# the first measurement (bounded leak of
# <= 2 years; FALSE leaves them NA)

# drop the entry bin from the risk set? Events fire there at roughly half the
# rate of other bins because entry occurs mid-bin (partial exposure). Leaving
# it in is fine; TRUE is the robustness check.
DROP_ENTRY_BIN <- FALSE

# ------------------------------------------------------------- the event map
# column -> event group.  "!remove" is dropped.

remap <- c(
  # --- cardiovascular ---
  "angina"      = "cvd",
  "heartattack" = "cvd",
  "chf"         = "cvd",
  "heart"       = "cvd",         # NOTE: supplies the majority of this group.
  # Check the user guide -- if it is a general
  # "any heart problem" item, the group is
  # broader than coronary heart disease and
  # should be labelled accordingly.
  "angin"       = "cvd",
  "hrtatt"      = "cvd",
  "hrtrhm"      = "!remove",     # arrhythmia: finding, not disease
  "hrtmr"       = "!remove",     # murmur: often benign / incidental
  "cvd"         = "!remove",     # derived from the above -> would double-count
  "stroke"      = "stroke",
  "strok"       = "stroke",
  # --- metabolic ---
  "hibp"        = "hypertension",
  "hchol"       = "!remove",     # 0% prevalent at entry + highest incidence:
  # an artefact of the wave it was introduced
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

STATIC <- c("sex", "edu", "wealth")     # bmi is added separately below

# ---------------------------------------------------------------- helpers

pmin_na <- function(m) {
  out <- suppressWarnings(apply(m, 1, min, na.rm = TRUE))
  out[is.infinite(out)] <- NA_real_
  out
}

src_of <- function(m, cols) {
  apply(m, 1, function(r)
    if (all(is.na(r))) NA_character_
    else cols[which.min(replace(r, is.na(r), Inf))])
}

# ------------------------------------------------------------------ 0. read

elsa_data <- read.csv(PATH_IN)
N0 <- nrow(elsa_data)
message("read ", N0, " rows from ", basename(PATH_IN))

# ------------------------------------------- 1. collapse columns into groups

remap  <- remap[remap != "!remove"]
remap  <- remap[names(remap) %in% names(elsa_data)]

groups <- sort(unique(unname(remap)))
groups <- c(setdiff(groups, "died"), "died")         # death last: it absorbs

for (g in groups) {
  cols <- names(remap)[remap == g]
  m    <- as.matrix(elsa_data[, cols, drop = FALSE])
  elsa_data[[paste0("age_", g)]] <- pmin_na(m)
  if (length(cols) > 1) elsa_data[[paste0("src_", g)]] <- src_of(m, cols)
}

EV      <- groups
age_col <- setNames(paste0("age_", EV), EV)
message("event groups (", length(EV), "): ", paste(EV, collapse = ", "))

# ---------------------------------------------------------- 2. child context

CHILD_MAP    <- CHILD_MAP[CHILD_MAP != "!remove"]
CHILD_MAP    <- CHILD_MAP[names(CHILD_MAP) %in% names(elsa_data)]
child_groups <- sort(unique(unname(CHILD_MAP)))

for (g in child_groups) {
  cols <- names(CHILD_MAP)[CHILD_MAP == g]
  elsa_data[[g]] <- as.integer(rowSums(!is.na(elsa_data[, cols, drop = FALSE])) > 0)
}
elsa_data$child_n <- rowSums(elsa_data[, child_groups, drop = FALSE])

# ------------------------------------------- 3. response waves from iwstat
#
# Harmonised HRS/ELSA coding:  1 = responded, alive        4 = alive, no resp
#                              5 = died this wave          6 = died earlier
#                              0 / 7 / 9 = not in sample / dropped / inap

iw_cols <- grep("^r[0-9]+iwstat$", names(elsa_data), value = TRUE)
iw_cols <- iw_cols[order(as.integer(gsub("\\D", "", iw_cols)))]

if (length(iw_cols) == 0)
  stop("no r*iwstat columns found. Censoring cannot be derived from the date ",
       "columns alone: they are populated for every wave regardless of ",
       "attendance, which would create immortal person-time. Merge the ",
       "harmonised response variables first.")

M <- as.matrix(elsa_data[, iw_cols]); storage.mode(M) <- "numeric"
resp <- M %in% RESP_CODES; dim(resp) <- dim(M)
W    <- ncol(resp)
any_resp <- rowSums(resp) > 0

elsa_data$first_wave  <- ifelse(any_resp, max.col(resp, ties.method = "first"), NA)
elsa_data$latest_wave <- ifelse(any_resp,
                                W + 1 - max.col(resp[, W:1, drop = FALSE],
                                                ties.method = "first"), NA)
deadm <- M == DEAD_CODE; deadm[is.na(deadm)] <- FALSE
elsa_data$death_wave <- ifelse(rowSums(deadm) > 0,
                               max.col(deadm, ties.method = "first"), NA)

cat("\n--- iwstat codes present ---\n"); print(table(M, useNA = "ifany"))
cat("\n--- first wave responded ---\n");  print(table(elsa_data$first_wave))
cat("\n--- last wave responded ---\n");   print(table(elsa_data$latest_wave))

# ------------------------------------------------------ 4. entry / exit ages

date_cols <- grep("^date[0-9]+$", names(elsa_data), value = TRUE)
date_cols <- date_cols[order(as.integer(sub("date", "", date_cols)))]
stopifnot(length(date_cols) > 0)

dates <- as.matrix(elsa_data[, date_cols])
W_d   <- ncol(dates)
if (W > W_d)
  message("NOTE: ", W, " iwstat waves but only ", W_d, " date columns. ",
          "Follow-up is capped at wave ", W_d, " -- ",
          sum(elsa_data$latest_wave > W_d, na.rm = TRUE),
          " people lose their last wave(s). Add date", W_d + 1,
          "..date", W, " to recover it.")

idx <- seq_len(nrow(elsa_data))
fw  <- pmin(elsa_data$first_wave,  W_d)
lw  <- pmin(elsa_data$latest_wave, W_d)

# --- age at death, best source first -----------------------------------
# NB: `age_died` already exists from the remap (the `died` column). Start from
# it and FILL the gaps -- do not overwrite. Every step is a full-length
# vector: an ifelse whose TEST is length 1 silently returns a length-1 result
# and recycles a single NA across the whole cohort.
age_died <- rep(NA_real_, nrow(elsa_data))
if ("age_died" %in% names(elsa_data)) age_died <- elsa_data$age_died
src_death <- ifelse(!is.na(age_died), "died_col", NA_character_)

if ("radage" %in% names(elsa_data)) {                     # ONS-linked, best
  use <- is.na(age_died) & !is.na(elsa_data$radage)
  age_died[use]  <- elsa_data$radage[use]
  src_death[use] <- "radage"
}
if ("yod" %in% names(elsa_data)) {
  use <- is.na(age_died) & !is.na(elsa_data$yod)
  age_died[use]  <- elsa_data$yod[use] - elsa_data$yob[use]
  src_death[use] <- "yod"
}
dw  <- pmin(elsa_data$death_wave, W_d)                    # iwstat == 5
use <- is.na(age_died) & !is.na(dw)
age_died[use]  <- dates[cbind(idx, dw)][use] - elsa_data$yob[use]
src_death[use] <- "iwstat5"

elsa_data$age_died <- age_died
cat("\n--- death ascertainment by source ---\n")
print(table(src_death, useNA = "no"))
message(sprintf("deaths resolved for %d people (%.1f%%)",
                sum(!is.na(age_died)), 100 * mean(!is.na(age_died))))

# DIFFERENTIAL FOLLOW-UP. Diseases are observable only while the person keeps
# responding: a diagnosis made after they stop is never reported. Death is
# known BEYOND that point -- iwstat records it at the wave where they are
# found dead, i.e. AFTER their last response -- so mortality follow-up runs
# longer than disease follow-up. Sharing one exit would either throw the
# deaths away (they fall past the grid) or credit the gap bins as disease-free
# when nobody observed them. Each event carries its own at-risk mask, so give
# death its own window.
elsa_data <- elsa_data %>%
  mutate(entry_age  = dates[cbind(idx, fw)] - yob,
         obs_end    = dates[cbind(idx, lw)] - yob,
         # start of the OBSERVATION GRID, not of the interview series.
         # Anything diagnosed before this is prevalent -- including diagnoses
         # between an early entry (partners enter from age 35) and AGE_MIN,
         # which would otherwise fall outside the grid and leave the person
         # at risk for a condition they already have.
         obs_start  = pmax(entry_age, AGE_MIN),
         exit_dis   = pmin(obs_end, ifelse(is.na(age_died), Inf, age_died)),
         exit_death = ifelse(is.na(age_died), exit_dis, age_died)) %>%
  filter(!is.na(entry_age), !is.na(exit_dis), exit_dis > entry_age)

# ------------------------------------------------------------ 5. age binning

n_bins <- as.integer((AGE_MAX - AGE_MIN) / BIN)
bin_of <- function(a) {
  b <- floor((a - AGE_MIN) / BIN) + 1L
  ifelse(is.na(a) | b < 1L | b > n_bins, NA_integer_, b)
}

elsa_data <- elsa_data %>%
  mutate(entry_bin      = bin_of(obs_start),
         exit_bin_dis   = bin_of(pmin(exit_dis,   AGE_MAX - 1e-6)),
         exit_bin_death = bin_of(pmin(exit_death, AGE_MAX - 1e-6)),
         exit_bin       = pmax(exit_bin_dis, exit_bin_death, na.rm = TRUE)) %>%
  filter(!is.na(entry_bin), !is.na(exit_bin_dis), exit_bin >= entry_bin)

message(sprintf(
  "n = %d people (%.0f%% of %d) | %d age bins (%g-%g, width %g) | entry age %.0f-%.0f | median disease follow-up %.1fy",
  nrow(elsa_data), 100 * nrow(elsa_data) / N0, N0, n_bins, AGE_MIN, AGE_MAX, BIN,
  min(elsa_data$entry_age), max(elsa_data$entry_age),
  median(elsa_data$exit_dis - elsa_data$obs_start)))
message(sprintf("  mortality follow-up extends past the last response for %d people (median gap %.1fy)",
                sum(elsa_data$exit_death > elsa_data$exit_dis + 1e-6),
                median((elsa_data$exit_death - elsa_data$exit_dis)[
                  elsa_data$exit_death > elsa_data$exit_dis + 1e-6])))


# ------------------------------------------------------ 6. IMPUTE MISSING COVARIATES


# a. Education
sum(is.na(elsa_data$edu)) #220
edu_median <- median(elsa_data$edu, na.rm = TRUE) #2
elsa_data$edu[is.na(elsa_data$edu)] <- edu_median


# b. Wealth
sum(is.na(elsa_data$wealth)) #22
wealth_median <- median(elsa_data$wealth, na.rm = TRUE) #3
elsa_data$wealth[is.na(elsa_data$wealth)] <- wealth_median

# c.bmi
sum(is.na(elsa_data$bmi_latest)) #3310
impute_df = elsa_data[c("idauniq", "yob","sex", "wealth","diabetes", "depression","bmi_latest", "edu" )]
mf = missForest::missForest(impute_df,ntree = 100,maxiter = 50)
elsa_data$bmi_latest = mf$ximp$bmi_latest


# ------------------------------------------------------ 6. SAVE elsa_data

write.csv(elsa_data, "C:/Users/dinab/Desktop/PhD Projects/Ensemble methods/GitHub_App/medicaljourneys/ELSA data/elsa_data.csv", row.names = FALSE)


# ------------------------------------------------------ 6. expand to the grid

long <- elsa_data %>%
  select(idauniq, entry_bin, exit_bin) %>%
  rowwise() %>%
  reframe(idauniq = idauniq, bin = seq(entry_bin, exit_bin)) %>%
  left_join(elsa_data, by = "idauniq") %>%
  mutate(age_mid = AGE_MIN + (bin - 0.5) * BIN)

# ------------------------------------------- 7. event / at-risk / prev flags

for (k in EV) {
  a  <- long[[age_col[[k]]]]
  kb <- bin_of(a)
  prevalent <- !is.na(a) & (a <= long$obs_start)     # present before the grid
  
  # death is followed past the last response; diseases are not
  last_bin <- if (k == "died") long$exit_bin_death else long$exit_bin_dis
  
  long[[paste0("event_", k)]] <- as.integer(!is.na(kb) & kb == long$bin &
                                              !prevalent & long$bin <= last_bin)
  long[[paste0("prev_",  k)]] <- as.integer(prevalent | (!is.na(kb) & long$bin > kb))
  at_risk <- !prevalent & (is.na(kb) | long$bin <= kb) & (long$bin <= last_bin)
  if (DROP_ENTRY_BIN) at_risk <- at_risk & (long$bin > long$entry_bin)
  long[[paste0("at_risk_", k)]] <- as.integer(at_risk)
}

# death absorbs: once dead, nothing else is at risk
dead_now <- long$prev_died == 1L
for (k in setdiff(EV, "died")) long[[paste0("at_risk_", k)]][dead_now] <- 0L

# --------------------------------------------------------------- 8. BMI
# Nurse-visit BMI is measured at waves 2/4/6/8. Carried forward to each age
# bin it is the one genuinely TIME-VARYING covariate available here, and it
# avoids the look-ahead leak of bmi_latest (measured at the LAST nurse visit,
# i.e. after most people's events).

bmi_cols <- paste0("r", BMI_WAVES, "mbmi")
have_bmi <- bmi_cols[bmi_cols %in% names(long)]

if (BMI_MODE == "timevarying" && length(have_bmi) > 0) {
  w_ok <- BMI_WAVES[bmi_cols %in% names(long)]
  bmi <- first_bmi <- rep(NA_real_, nrow(long))
  for (j in seq_along(w_ok)) {                       # ascending -> LOCF
    v  <- long[[paste0("r", w_ok[j], "mbmi")]]
    dc <- paste0("date", w_ok[j])
    ag <- if (dc %in% names(long)) long[[dc]] - long$yob else rep(NA_real_, nrow(long))
    ok <- !is.na(v) & !is.na(ag) & ag <= long$age_mid
    bmi <- ifelse(ok, v, bmi)
    first_bmi <- ifelse(is.na(first_bmi) & !is.na(v), v, first_bmi)
  }
  if (BMI_BACKFILL) bmi <- ifelse(is.na(bmi), first_bmi, bmi)
  long$bmi <- bmi
  message(sprintf("BMI: time-varying from waves %s | %.1f%% of person-bins have a value",
                  paste(w_ok, collapse = "/"), 100 * mean(!is.na(long$bmi))))
} else if (BMI_MODE == "first" && length(have_bmi) > 0) {
  long$bmi <- do.call(coalesce, unname(long[, have_bmi]))
  message("BMI: first available nurse visit, fixed at baseline")
} else {
  long$bmi <- long$bmi_latest
  warning("BMI_MODE='latest': bmi_latest is measured at the LAST nurse visit, ",
          "so for most people it POSTDATES their events. This is a look-ahead ",
          "leak and will inflate discrimination.")
}

# --------------------------------------------------------- 9. assemble & fill

COVARS <- c(STATIC, "bmi")

# impute at PERSON level (a person-bin mean would weight by follow-up length)
per_person <- long %>% group_by(idauniq) %>% slice_min(bin, n = 1) %>% ungroup()
for (cc in COVARS) {
  if (!cc %in% names(long)) next
  n_na <- sum(is.na(long[[cc]]))
  if (n_na > 0) {
    fill <- median(per_person[[cc]], na.rm = TRUE)
    long[[paste0(cc, "_miss")]] <- as.integer(is.na(long[[cc]]))
    long[[cc]] <- ifelse(is.na(long[[cc]]), fill, long[[cc]])
    message(sprintf("  imputed %s with %.2f (%.1f%% of person-bins); kept %s_miss",
                    cc, fill, 100 * n_na / nrow(long), cc))
  }
}

long <- long %>% mutate(age_c = (age_mid - 70) / 10)

keep <- c("idauniq", "bin", "age_mid", "age_c", "entry_bin", "exit_bin",
          "first_wave", "latest_wave",
          COVARS, grep("_miss$", names(long), value = TRUE),
          child_groups, "child_n",
          grep("^src_", names(long), value = TRUE),
          paste0("event_", EV), paste0("prev_", EV), paste0("at_risk_", EV))

long <- long %>% select(any_of(keep)) %>% arrange(idauniq, bin)

# ----------------------------------------------------------------- 10. checks

cat("\n--- incident events per group (left truncation + iwstat censoring) ---\n")
ev_counts <- long %>%
  summarise(across(starts_with("event_"), sum)) %>%
  pivot_longer(everything(), names_to = "event", values_to = "n") %>%
  mutate(event = sub("event_", "", event)) %>%
  arrange(n)
print(ev_counts, n = Inf)

cat("\n--- person-bins AT RISK per group (death's window is longer) ---\n")
print(long %>% summarise(across(starts_with("at_risk_"), sum)) %>%
        pivot_longer(everything(), names_to = "event", values_to = "at_risk") %>%
        mutate(event = sub("at_risk_", "", event)) %>%
        left_join(ev_counts, by = "event") %>%
        mutate(rate_per_1000 = round(1000 * n / at_risk, 1)) %>%
        arrange(desc(at_risk)), n = Inf)

thin <- ev_counts %>% filter(n < MIN_EVENTS)
if (nrow(thin))
  message("\nTOO THIN for a per-age-bin hazard (< ", MIN_EVENTS, " events): ",
          paste(thin$event, collapse = ", "), "\n  -> merge or drop these.")

cat("\n--- prevalent at entry (%) ---\n")
print(long %>% group_by(idauniq) %>% slice_min(bin, n = 1) %>% ungroup() %>%
        summarise(across(starts_with("prev_"), ~ round(100 * mean(.x), 1))) %>%
        pivot_longer(everything(), names_to = "event", values_to = "pct") %>%
        mutate(event = sub("prev_", "", event)) %>% arrange(desc(pct)), n = Inf)

cat("\n--- person-bins per age bin ---\n")
print(long %>% count(bin, age_mid), n = n_bins)

cat("\n--- follow-up: bins per person ---\n")
print(table(long %>% count(idauniq) %>% pull(n)))

cat("\n--- events in the ENTRY bin (entry is mid-bin, so ~half the usual rate) ---\n")
for (k in EV) {
  tot <- sum(long[[paste0("event_", k)]])
  if (tot == 0) next
  ne  <- sum(long[[paste0("event_", k)]] == 1 & long$bin == long$entry_bin)
  cat(sprintf("%-20s %5d / %5d  (%4.1f%%)\n", k, ne, tot, 100 * ne / tot))
}

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

for (k in EV) {
  bad <- sum(long[[paste0("event_", k)]] == 1 & long[[paste0("at_risk_", k)]] == 0)
  if (bad) warning(sprintf("%s: %d events outside the risk set", k, bad))
}

write.csv(long, PATH_OUT, row.names = FALSE)
message("\nwritten: ", basename(PATH_OUT), "  (", nrow(long), " person-bins, ",
        length(unique(long$idauniq)), " people)")

dd <- elsa_data$age_died
cat("resolved in the modelling cohort:", sum(!is.na(dd)), "\n",
    "  after AGE_MAX:", sum(dd > AGE_MAX, na.rm=TRUE), "\n",
    "  before obs_start:", sum(dd <= elsa_data$obs_start, na.rm=TRUE), "\n",
    "  fired:", 1378, "\n")


