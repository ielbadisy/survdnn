# basetable/base-R replacements for purrr::map_dfr()/purrr::imap_dfr(): map
# (or index-map) .f over .x and row-bind the results, matching purrr's
# silent drop of NULL results (basetable::rbindfill() already drops NULLs).
.map_dfr <- function(.x, .f, ...) {
  tibble::as_tibble(basetable::rbindfill(basetable::map(.x, .f, ...), fill = TRUE))
}

.imap_dfr <- function(.x, .f) {
  idx <- if (!is.null(names(.x))) names(.x) else seq_along(.x)
  tibble::as_tibble(basetable::rbindfill(basetable::traverse(list(.x, idx), .f), fill = TRUE))
}

# .map_dfr()'s list-column-safe counterpart: basetable::rbindfill() coerces
# list-columns (e.g. the `hidden` per-config vector of hidden-layer sizes)
# to NA instead of preserving them, unlike base rbind()/do.call(rbind, ...).
# Use this wherever a mapped result carries a list-column.
.map_rbind <- function(.x, .f, ...) {
  do.call(rbind, basetable::map(.x, .f, ...))
}

# tidyr::crossing(!!!param_grid) replacement: the deduplicated full
# cross-join of every param_grid entry, with list-valued entries (e.g.
# survdnn's `hidden`, a per-config vector of hidden-layer sizes) preserved
# as a list-column instead of atomic-vector expansion. tidyr::crossing()
# varies its last argument fastest; expand.grid() varies its first argument
# fastest, hence the reversed expand.grid() call followed by reordering
# columns back to param_grid's original order.
.crossing_grid <- function(param_grid) {
  uniq <- lapply(param_grid, function(x) {
    if (is.list(x)) {
      x[!duplicated(lapply(x, function(e) paste(e, collapse = "_")))]
    } else {
      sort(unique(x))
    }
  })
  nms <- names(uniq)
  idx_lists <- lapply(uniq, seq_along)
  idx_grid <- do.call(expand.grid, c(rev(idx_lists), list(KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)))
  idx_grid <- idx_grid[, nms, drop = FALSE]
  out <- as.data.frame(lapply(nms, function(nm) {
    vals <- uniq[[nm]][idx_grid[[nm]]]
    if (is.list(uniq[[nm]])) I(vals) else vals
  }), stringsAsFactors = FALSE)
  names(out) <- nms
  tibble::as_tibble(out)
}

# tidyr::pivot_longer() replacement for the simple "melt these columns, keep
# the rest as id columns" case.
.pivot_longer_simple <- function(df, cols, names_to, values_to) {
  id_cols <- setdiff(names(df), cols)
  n <- nrow(df)
  k <- length(cols)
  row_idx <- rep(seq_len(n), each = k)
  id_part <- df[row_idx, id_cols, drop = FALSE]
  name_part <- rep(cols, times = n)
  value_cols <- df[, cols, drop = FALSE]
  value_part <- do.call(c, lapply(seq_len(n), function(i) unlist(value_cols[i, ], use.names = FALSE)))
  out <- cbind(id_part, stats::setNames(
    data.frame(name_part, value_part, stringsAsFactors = FALSE),
    c(names_to, values_to)
  ))
  rownames(out) <- NULL
  tibble::as_tibble(out)
}
