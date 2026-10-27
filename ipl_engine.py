"""IPL engine: player ratings, Playing-11 winner model, Player-of-the-Match model
aur live (half-match) models. Ye file app.py ke saath same folder me rakho.

Zaroori files: matches.csv aur deliveries.csv (Kaggle IPL dataset).
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TEAM_RENAMES = {
    "Delhi Daredevils": "Delhi Capitals",
    "Deccan Chargers": "Sunrisers Hyderabad",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
}
# In dismissals ka credit bowler ko milta hai
BOWLER_WICKETS = {"caught", "bowled", "lbw", "caught and bowled", "stumped", "hit wicket"}

OLD_CAT = ["team1", "team2", "venue", "toss_decision", "toss_by_team1"]  # sirf comparison ke liye
CAT = ["toss_decision", "toss_by_team1"]
NUM = ["d_bat", "d_wk", "d_econ", "d_venue"]
LIVE2 = ["runs_left", "balls_left", "wickets_left", "crr", "rrr", "target"]
LIVE1 = ["score", "wickets_left", "balls_left", "crr", "venue_avg"]


def _norm_team(s):
    return s.astype(str).str.strip().replace(TEAM_RENAMES)


def _make_X(df):
    X = pd.DataFrame(
        {
            "team1": df["team1"].values,
            "team2": df["team2"].values,
            "venue": df["venue"].values,
            "toss_decision": df["toss_decision"].values,
            "toss_by_team1": (df["toss_winner"].values == df["team1"].values).astype(int).astype(str),
        }
    )
    for c in NUM:
        X[c] = df[c].values
    return X


def _swap(df):
    s = df.copy()
    s["team1"], s["team2"] = df["team2"], df["team1"]
    for c in NUM:
        s[c] = -df[c]
    return s


def _win_pipe(cat, num, C):
    tr = [("cat", OneHotEncoder(handle_unknown="ignore"), cat)]
    if num:
        tr.append(("num", StandardScaler(), num))
    return Pipeline([("prep", ColumnTransformer(tr)), ("clf", LogisticRegression(C=C, max_iter=3000))])


def _fit_win(train_df, cat=CAT, num=NUM, C=0.05):
    aug = pd.concat([train_df, _swap(train_df)])
    y = (aug["winner"] == aug["team1"]).astype(int)
    pipe = _win_pipe(cat, num, C)
    pipe.fit(_make_X(aug), y)
    return pipe


def _win_p(pipe, df):
    """Team1 ke jeetne ki probability, dono orders ka average."""
    p1 = pipe.predict_proba(_make_X(df))[:, 1]
    p2 = pipe.predict_proba(_make_X(_swap(df)))[:, 1]
    return (p1 + 1 - p2) / 2


def _cv_win(dm, cat, num, C):
    """Time-series CV: purane matches par train, naye par test (5 folds). (accuracy, log-loss)"""
    n = len(dm)
    accs, lls = [], []
    for a, b in [(.5, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.0)]:
        tr, te = dm.iloc[: int(n * a)], dm.iloc[int(n * a): int(n * b)]
        p = np.clip(_win_p(_fit_win(tr, cat, num, C), te), 1e-6, 1 - 1e-6)
        y = (te["winner"] == te["team1"]).astype(int).to_numpy()
        accs.append(np.mean((p >= 0.5) == y))
        lls.append(log_loss(y, p))
    return float(np.mean(accs)), float(np.mean(lls))


def _potm_X(F, won):
    X = pd.DataFrame(
        {
            "bat_dev": F["bat_dev"].values,
            "wk_dev": F["wk_dev"].values,
            "econ_dev": F["econ_dev"].values,
            "sr_dev": F["sr_dev"].values,
            "bat_share": F["bat_share"].values,
            "bowl_share": F["bowl_share"].values,
            "log_exp": np.log1p(F["exp"].values),
            "won": np.asarray(won, dtype=float),
        }
    )
    X["won_bat"] = X["won"] * X["bat_dev"]
    X["won_wk"] = X["won"] * X["wk_dev"]
    return X


def _potm_pipe():
    return Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(C=0.3, max_iter=3000))])


class IPLEngine:
    def __init__(self, matches_raw, deliv_raw):
        self.metrics = {}
        self._prepare(matches_raw, deliv_raw)
        self._player_table()
        self._win_models()
        self._potm_model()
        self._live_models()

    # ------------------------------------------------------------ prepare
    def _prepare(self, matches_raw, deliv_raw):
        m = matches_raw.copy()
        m.columns = [str(c).strip() for c in m.columns]
        need = ["id", "date", "player_of_match", "venue", "team1", "team2",
                "toss_winner", "toss_decision", "winner"]
        miss = [c for c in need if c not in m.columns]
        if miss:
            raise ValueError("matches.csv me ye columns chahiye: " + ", ".join(miss))
        for c in ["team1", "team2", "toss_winner", "winner"]:
            m[c] = _norm_team(m[c])
        m["venue"] = m["venue"].astype(str).str.split(",").str[0].str.strip()
        m["toss_decision"] = m["toss_decision"].astype(str).str.strip().str.lower()
        m["date"] = pd.to_datetime(m["date"], errors="coerce")
        m = m.dropna(subset=["date"]).sort_values(["date", "id"]).reset_index(drop=True)
        m["decisive"] = (m["winner"] == m["team1"]) | (m["winner"] == m["team2"])
        self.m = m

        d = deliv_raw.copy()
        d.columns = [str(c).strip() for c in d.columns]
        dneed = ["match_id", "inning", "batting_team", "bowling_team", "batter", "bowler",
                 "non_striker", "batsman_runs", "total_runs", "extras_type", "is_wicket",
                 "player_dismissed", "dismissal_kind"]
        miss = [c for c in dneed if c not in d.columns]
        if miss:
            raise ValueError("deliveries.csv me ye columns chahiye: " + ", ".join(miss))
        d["batting_team"] = _norm_team(d["batting_team"])
        d["bowling_team"] = _norm_team(d["bowling_team"])
        d["extras_type"] = d["extras_type"].fillna("")
        d["dismissal_kind"] = d["dismissal_kind"].fillna("")
        d["player_dismissed"] = d["player_dismissed"].fillna("")
        d["is_wicket"] = d["is_wicket"].fillna(0).astype(int)
        self.d = d

    # ------------------------------------------------------------ players
    def _ratings(self, df, pre):
        P = self.P
        runs, balls = df[pre + "runs"], df[pre + "balls"]
        bi, wk = df[pre + "bat_inn"], df[pre + "wkts"]
        bb, br = df[pre + "bowl_balls"], df[pre + "bowl_runs"]
        boi, mt = df[pre + "bowl_inn"], df[pre + "matches"]
        out = pd.DataFrame(index=df.index)
        out["bat_rpi"] = (runs + 8 * P["rpi"]) / (bi + 8)
        out["bat_sr"] = 100 * (runs + 60 * P["rpb"]) / (balls + 60)
        out["bowl_wpi"] = (wk + 8 * P["wpi"]) / (boi + 8)
        out["bowl_econ"] = 6 * (br + 60 * P["bpb"]) / (bb + 60)
        out["bat_dev"] = out["bat_rpi"] - P["rpi"]
        out["wk_dev"] = out["bowl_wpi"] - P["wpi"]
        out["econ_dev"] = 6 * P["bpb"] - out["bowl_econ"]
        out["sr_dev"] = out["bat_sr"] - 100 * P["rpb"]
        out["exp"] = mt
        safe = np.maximum(mt, 1)
        out["bat_share"] = bi / safe
        out["bowl_share"] = boi / safe
        return out

    def _player_table(self):
        d, m = self.d, self.m
        faced = d[d["extras_type"] != "wides"]
        bat = (faced.groupby(["match_id", "batter"])
               .agg(runs=("batsman_runs", "sum"), balls=("batsman_runs", "size"))
               .reset_index().rename(columns={"batter": "player"}))
        outs = d[(d["is_wicket"] == 1) & (d["player_dismissed"] != "")
                 & (d["dismissal_kind"] != "retired hurt")]
        outs = (outs.groupby(["match_id", "player_dismissed"]).size().reset_index(name="out")
                .rename(columns={"player_dismissed": "player"}))
        outs["out"] = 1

        bd = d[["match_id", "bowler", "extras_type", "total_runs", "is_wicket", "dismissal_kind"]].copy()
        bd["legal"] = (~bd["extras_type"].isin(["wides", "noballs"])).astype(int)
        bd["conc"] = np.where(bd["extras_type"].isin(["byes", "legbyes", "penalty"]), 0, bd["total_runs"])
        bd["wk"] = ((bd["is_wicket"] == 1) & bd["dismissal_kind"].isin(BOWLER_WICKETS)).astype(int)
        bowl = (bd.groupby(["match_id", "bowler"])
                .agg(bowl_balls=("legal", "sum"), bowl_runs=("conc", "sum"), wkts=("wk", "sum"))
                .reset_index().rename(columns={"bowler": "player"}))

        parts = []
        for pcol, tcol in [("batter", "batting_team"), ("non_striker", "batting_team"), ("bowler", "bowling_team")]:
            parts.append(d[["match_id", pcol, tcol]].rename(columns={pcol: "player", tcol: "team"}))
        app = pd.concat(parts).drop_duplicates(["match_id", "player"])

        pm = (app.merge(bat, on=["match_id", "player"], how="left")
              .merge(outs, on=["match_id", "player"], how="left")
              .merge(bowl, on=["match_id", "player"], how="left"))
        for c in ["runs", "balls", "out", "bowl_balls", "bowl_runs", "wkts"]:
            pm[c] = pm[c].fillna(0)
        pm["bat_inn"] = (pm["balls"] > 0).astype(int)
        pm["bowl_inn"] = (pm["bowl_balls"] > 0).astype(int)
        pm = pm.merge(m[["id", "date", "winner", "player_of_match", "decisive"]],
                      left_on="match_id", right_on="id", how="inner").drop(columns="id")
        pm = pm.sort_values(["date", "match_id"]).reset_index(drop=True)

        g = pm.groupby("player")
        for s in ["runs", "balls", "bat_inn", "wkts", "bowl_balls", "bowl_runs", "bowl_inn"]:
            pm["pre_" + s] = g[s].cumsum() - pm[s]
        pm["pre_matches"] = g.cumcount()

        self.P = {
            "rpi": pm["runs"].sum() / max(pm["bat_inn"].sum(), 1),
            "rpb": pm["runs"].sum() / max(pm["balls"].sum(), 1),
            "wpi": pm["wkts"].sum() / max(pm["bowl_inn"].sum(), 1),
            "bpb": pm["bowl_runs"].sum() / max(pm["bowl_balls"].sum(), 1),
        }
        rt = self._ratings(pm, "pre_")
        self.pm = pd.concat([pm, rt], axis=1)

        # Abhi ki (poore data ke baad) rating - UI me chune gaye players ke liye
        cur = pm.groupby("player").agg(
            cur_runs=("runs", "sum"), cur_balls=("balls", "sum"), cur_bat_inn=("bat_inn", "sum"),
            cur_wkts=("wkts", "sum"), cur_bowl_balls=("bowl_balls", "sum"),
            cur_bowl_runs=("bowl_runs", "sum"), cur_bowl_inn=("bowl_inn", "sum"),
            cur_matches=("match_id", "size"),
        )
        self.cur_raw = cur
        self.cur = self._ratings(cur, "cur_")
        self.players = sorted(cur.index.tolist())

        # Squad pools: har team ke haal ke players
        last = pm["date"].max()
        self.pools, self.defaults = {}, {}
        recent2 = pm[pm["date"] >= last - pd.Timedelta(days=730)]
        recent1 = pm[pm["date"] >= last - pd.Timedelta(days=300)]
        for team in sorted(pm["team"].unique()):
            c2 = recent2[recent2["team"] == team]["player"].value_counts()
            c1 = recent1[recent1["team"] == team]["player"].value_counts()
            allc = pm[pm["team"] == team]["player"].value_counts()
            pool = c2.index.tolist() or allc.index.tolist()
            self.pools[team] = pool
            self.defaults[team] = (c1.index.tolist() or pool)[:11]
        self.last_date = last

    def player_options(self, team):
        pool = self.pools.get(team, [])
        rest = [p for p in self.players if p not in set(pool)]
        return pool + rest

    def default_xi(self, team):
        return self.defaults.get(team, [])[:11]

    def player_table(self, names):
        raw = self.cur_raw.reindex(names).fillna(0)
        r = self._ratings(raw, "cur_")
        out = pd.DataFrame({
            "Player": names,
            "Matches": raw["cur_matches"].astype(int).values,
            "Runs": raw["cur_runs"].astype(int).values,
            "Bat avg/inn": r["bat_rpi"].round(1).values,
            "Strike rate": r["bat_sr"].round(1).values,
            "Wickets": raw["cur_wkts"].astype(int).values,
            "Economy": np.where(raw["cur_bowl_balls"].values > 0, r["bowl_econ"].round(2).values, np.nan),
        })
        return out

    # ------------------------------------------------------------ winner models
    def _win_models(self):
        m, pm = self.m, self.pm
        tm = (pm.groupby(["match_id", "team"])
              .agg(bat=("bat_dev", "sum"), wk=("wk_dev", "sum"), econ=("econ_dev", "sum"))
              .reset_index().set_index(["match_id", "team"]))
        dm = m[m["decisive"]].copy().reset_index(drop=True)
        f1 = tm.reindex(pd.MultiIndex.from_arrays([dm["id"], dm["team1"]])).fillna(0).to_numpy()
        f2 = tm.reindex(pd.MultiIndex.from_arrays([dm["id"], dm["team2"]])).fillna(0).to_numpy()
        dm["d_bat"], dm["d_wk"], dm["d_econ"] = f1[:, 0] - f2[:, 0], f1[:, 1] - f2[:, 1], f1[:, 2] - f2[:, 2]

        # Conditions: venue par pehle batting karne wali team ka record (sirf pehle ke matches se)
        other = np.where(dm["toss_winner"] == dm["team1"], dm["team2"], dm["team1"])
        dm["bat_first"] = np.where(dm["toss_decision"] == "bat", dm["toss_winner"], other)
        dm["bf_won"] = (dm["winner"] == dm["bat_first"]).astype(int)
        gv = dm.groupby("venue")
        pre_w = gv["bf_won"].cumsum() - dm["bf_won"]
        pre_n = gv.cumcount()
        dm["venue_adv"] = (pre_w + 5) / (pre_n + 10) - 0.5
        dm["d_venue"] = np.where(dm["bat_first"] == dm["team1"], 1, -1) * dm["venue_adv"] * 10
        self.dm = dm

        # Honest evaluation: purane par train, naye par test
        acc, ll = _cv_win(dm, CAT, NUM, 0.05)
        b_acc, b_ll = _cv_win(dm, OLD_CAT, [], 0.1)
        self.metrics.update(win_cv_acc=acc, win_cv_ll=ll, win_base_acc=b_acc, win_base_ll=b_ll,
                            win_coin_ll=float(np.log(2)))
        cut = int(len(dm) * 0.8)
        te = dm.iloc[cut:]
        self._p_te = pd.Series(_win_p(_fit_win(dm.iloc[:cut]), te), index=te["id"].values)
        self.win_xi = _fit_win(dm)

        # Venue ki jaankari (conditions)
        total1 = self.d[self.d["inning"] == 1].groupby("match_id")["total_runs"].sum()
        dm["total1"] = dm["id"].map(total1)
        vs = dm.groupby("venue").agg(matches=("id", "size"), avg_first=("total1", "mean"),
                                     bat_first_win=("bf_won", "mean"), bf_wins=("bf_won", "sum"))
        self.venue_stats = vs
        self.global_first = float(dm["total1"].mean())
        self.global_bat_first_win = float(dm["bf_won"].mean())

    def venue_info(self, venue):
        if venue in self.venue_stats.index:
            r = self.venue_stats.loc[venue]
            return int(r["matches"]), float(r["avg_first"]), float(r["bat_first_win"]) * 100
        return 0, self.global_first, self.global_bat_first_win * 100

    def xi_strength(self, players):
        r = self.cur.reindex(players)
        return (float(r["bat_dev"].fillna(0).sum()),
                float(r["wk_dev"].fillna(0).sum()),
                float(r["econ_dev"].fillna(0).sum()))

    def win_prob(self, a, b, xi_a, xi_b, venue, toss_winner, decision):
        """Team A ke jeetne ki probability (Playing 11 + toss + venue ke hisaab se)."""
        sa, sb = self.xi_strength(xi_a), self.xi_strength(xi_b)
        other = b if toss_winner == a else a
        bat_first = toss_winner if decision == "bat" else other
        if venue in self.venue_stats.index:
            vr = self.venue_stats.loc[venue]
            adv = (vr["bf_wins"] + 5) / (vr["matches"] + 10) - 0.5
        else:
            adv = 0.0
        d_venue = (1 if bat_first == a else -1) * adv * 10
        row = pd.DataFrame([{
            "team1": a, "team2": b, "venue": venue, "toss_winner": toss_winner,
            "toss_decision": decision, "winner": a,
            "d_bat": sa[0] - sb[0], "d_wk": sa[1] - sb[1], "d_econ": sa[2] - sb[2],
            "d_venue": d_venue,
        }])
        return float(_win_p(self.win_xi, row)[0]), sa, sb

    # ------------------------------------------------------------ Player of the Match
    def _potm_model(self):
        pm = self.pm
        dec = pm[pm["decisive"]].copy()
        dec["won"] = (dec["team"] == dec["winner"]).astype(int)
        dec["y"] = (dec["player"] == dec["player_of_match"]).astype(int)
        has = dec.groupby("match_id")["y"].transform("sum") > 0  # POTM jo dataset me khela hi nahi (rare) hata do
        dec = dec[has].reset_index(drop=True)
        X = _potm_X(dec, dec["won"].values)

        order = self.dm["id"].tolist()
        cut_ids = set(order[: int(len(order) * 0.8)])
        trm = dec["match_id"].isin(cut_ids).to_numpy()
        ev = _potm_pipe().fit(X[trm], dec.loc[trm, "y"])

        # Test: model ke khud ke pre-match win probability ke saath
        te = dec[~trm].copy()
        pw_team1 = self._p_te
        t1 = self.dm.set_index("id")["team1"]
        pw_row = te["match_id"].map(pw_team1)
        pw_row = np.where(te["team"].values == te["match_id"].map(t1).values, pw_row, 1 - pw_row)
        Fte = te
        p_won = ev.predict_proba(_potm_X(Fte, np.ones(len(te))))[:, 1]
        p_lost = ev.predict_proba(_potm_X(Fte, np.zeros(len(te))))[:, 1]
        te["score"] = pw_row * p_won + (1 - pw_row) * p_lost
        top1 = top3 = r1 = r3 = n = 0
        for _, g in te.groupby("match_id"):
            g = g.sort_values("score", ascending=False)
            n += 1
            top1 += int(g["y"].iloc[0] == 1)
            top3 += int(g["y"].iloc[:3].sum() > 0)
            r1 += 1 / len(g)
            r3 += min(3, len(g)) / len(g)
        self.metrics["potm_top1"] = top1 / max(n, 1)
        self.metrics["potm_top3"] = top3 / max(n, 1)
        self.metrics["potm_rand1"] = r1 / max(n, 1)
        self.metrics["potm_rand3"] = r3 / max(n, 1)
        self.potm = _potm_pipe().fit(X, dec["y"])

    def potm_predict(self, a, b, xi_a, xi_b, pa):
        """Dono Playing 11 ke har player ke Player-of-the-Match banne ki probability."""
        names = list(xi_a) + list(xi_b)
        teams = [a] * len(xi_a) + [b] * len(xi_b)
        pw = np.array([pa] * len(xi_a) + [1 - pa] * len(xi_b))
        raw = self.cur_raw.reindex(names).fillna(0)
        F = self._ratings(raw, "cur_")
        p_won = self.potm.predict_proba(_potm_X(F, np.ones(len(names))))[:, 1]
        p_lost = self.potm.predict_proba(_potm_X(F, np.zeros(len(names))))[:, 1]
        p = pw * p_won + (1 - pw) * p_lost
        p = p / p.sum()
        out = self.player_table(names)
        out.insert(1, "Team", teams)
        out["POTM %"] = (p * 100).round(1)
        return out.sort_values("POTM %", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------ live models
    def _live_models(self):
        d, m = self.d, self.m
        ok = m[m["decisive"]].copy()
        if "method" in ok.columns:
            ok = ok[ok["method"].isna()]
        if "target_overs" in ok.columns:
            to = pd.to_numeric(ok["target_overs"], errors="coerce")
            ok = ok[to.isna() | (to == 20)]
        ok = ok.reset_index(drop=True)
        ids = ok["id"].tolist()
        test_ids = set(ids[int(len(ids) * 0.8):])

        dd = d[d["match_id"].isin(set(ids)) & d["inning"].isin([1, 2])].copy()
        dd["legal"] = (~dd["extras_type"].isin(["wides", "noballs"])).astype(int)
        dd["wk_lost"] = ((dd["is_wicket"] == 1) & (dd["dismissal_kind"] != "retired hurt")).astype(int)
        g = dd.groupby(["match_id", "inning"], sort=False)
        dd["score"] = g["total_runs"].cumsum()
        dd["wkts"] = g["wk_lost"].cumsum()
        dd["balls"] = g["legal"].cumsum()
        dd = dd[dd["legal"] == 1]

        total1 = d[(d["inning"] == 1) & d["match_id"].isin(set(ids))].groupby("match_id")["total_runs"].sum()
        win = ok.set_index("id")["winner"]
        ven = ok.set_index("id")["venue"]

        # ---- 2nd innings (chase)
        s2 = dd[dd["inning"] == 2].copy()
        s2["target"] = s2["match_id"].map(total1) + 1
        s2["runs_left"] = s2["target"] - s2["score"]
        s2["balls_left"] = 120 - s2["balls"]
        s2["wickets_left"] = 10 - s2["wkts"]
        s2 = s2[(s2["runs_left"] > 0) & (s2["balls_left"] > 0) & (s2["wickets_left"] > 0)].copy()
        s2["crr"] = s2["score"] * 6 / s2["balls"]
        s2["rrr"] = s2["runs_left"] * 6 / s2["balls_left"]
        y2 = (s2["batting_team"] == s2["match_id"].map(win)).astype(int)
        te2 = s2["match_id"].isin(test_ids).to_numpy()

        def clf():
            return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=150,
                                                  l2_regularization=1.0, random_state=0)

        c2 = clf().fit(s2.loc[~te2, LIVE2], y2[~te2])
        p2 = c2.predict_proba(s2.loc[te2, LIVE2])[:, 1]
        self.metrics["live2_acc"] = float(accuracy_score(y2[te2], p2 >= 0.5))
        self.metrics["live2_auc"] = float(roc_auc_score(y2[te2], p2))
        self.live2 = clf().fit(s2[LIVE2], y2)

        # ---- 1st innings
        vg = total1.groupby(ven.reindex(total1.index).values).agg(["mean", "count"])
        gmean = float(total1.mean())
        self.v_avg = ((vg["mean"] * vg["count"] + gmean * 5) / (vg["count"] + 5)).to_dict()
        self.v_avg_default = gmean

        s1 = dd[dd["inning"] == 1].copy()
        s1["wickets_left"] = 10 - s1["wkts"]
        s1["balls_left"] = 120 - s1["balls"]
        s1 = s1[(s1["balls_left"] > 0) & (s1["wickets_left"] > 0)].copy()
        s1["crr"] = s1["score"] * 6 / s1["balls"]
        s1["venue_avg"] = s1["match_id"].map(ven).map(self.v_avg).fillna(gmean)
        y1 = (s1["batting_team"] == s1["match_id"].map(win)).astype(int)
        r1 = s1["match_id"].map(total1)
        te1 = s1["match_id"].isin(test_ids).to_numpy()

        c1 = clf().fit(s1.loc[~te1, LIVE1], y1[~te1])
        pp1 = c1.predict_proba(s1.loc[te1, LIVE1])[:, 1]
        self.metrics["live1_acc"] = float(accuracy_score(y1[te1], pp1 >= 0.5))
        self.metrics["live1_auc"] = float(roc_auc_score(y1[te1], pp1))

        def reg():
            return HistGradientBoostingRegressor(max_depth=4, learning_rate=0.08, max_iter=150,
                                                 l2_regularization=1.0, random_state=0)

        rg = reg().fit(s1.loc[~te1, LIVE1], r1[~te1])
        pr = rg.predict(s1.loc[te1, LIVE1])
        self.metrics["live1_mae"] = float(mean_absolute_error(r1[te1], pr))
        # Har over par error (projection kitni sahi hai) - range dikhane ke liye
        ov = (s1.loc[te1, "balls"].values - 1) // 6
        err = np.abs(r1[te1].values - pr)
        self.mae1_by_over = {int(o): float(err[ov == o].mean()) for o in np.unique(ov)}
        self.live1_clf = clf().fit(s1[LIVE1], y1)
        self.live1_reg = reg().fit(s1[LIVE1], r1)
        self.metrics["live_test_matches"] = len(test_ids)

    def live_chase(self, target, score, wkts, balls):
        """Chase karne wali team ke jeetne ki probability."""
        runs_left = target - score
        balls_left = 120 - balls
        if runs_left <= 0:
            return 1.0
        if wkts >= 10 or balls_left <= 0:
            return 0.0
        X = pd.DataFrame([{
            "runs_left": runs_left, "balls_left": balls_left, "wickets_left": 10 - wkts,
            "crr": score * 6 / balls if balls > 0 else 0.0,
            "rrr": runs_left * 6 / balls_left, "target": target,
        }])[LIVE2]
        return float(self.live2.predict_proba(X)[0, 1])

    def live_first(self, venue, score, wkts, balls):
        """(projected final score, +- range, pehle batting karne wali team ke jeetne ki probability)"""
        va = self.v_avg.get(venue, self.v_avg_default)
        X = pd.DataFrame([{
            "score": score, "wickets_left": 10 - wkts, "balls_left": 120 - balls,
            "crr": score * 6 / balls if balls > 0 else 0.0, "venue_avg": va,
        }])[LIVE1]
        proj = float(self.live1_reg.predict(X)[0])
        proj = max(proj, float(score))
        over = int(max(balls - 1, 0) // 6)
        keys = sorted(self.mae1_by_over)
        k = min(keys, key=lambda o: abs(o - over)) if keys else 0
        mae = self.mae1_by_over.get(k, self.metrics.get("live1_mae", 20.0))
        p = float(self.live1_clf.predict_proba(X)[0, 1])
        return proj, mae, p
