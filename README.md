## Final: Spain vs Argentina

Before the match, v1.6.5 gave:

- Spain win: 53.16%
- Draw: 33.26%
- Argentina win: 13.58%
- Most likely score: Spain 1–0 (17.78%)

The match was 0–0 after 90 minutes, before Spain scored in the 106th
minute and won 1–0.

So the 90-minute prediction was wrong. The model favored Spain and the
final score eventually became 1–0, but the goal came in extra time, which
this version does not model. I am keeping it as a miss rather than stretching
the evaluation rule after seeing the result.

The fitted rho also reached its 0.20 ceiling again. I initially introduced
the ceiling to keep the Dixon–Coles adjustment mathematically safe, but
repeatedly hitting it suggests that the model is asking the correction term
to explain more than it should.
