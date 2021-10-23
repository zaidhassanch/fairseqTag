

import fairseq.logging.metrics as metrics

with metrics.aggregate("train"):
    for i in range(10):
        print(i)
        metrics.log_scalar("loss1", i/10.0)

        for j in range(100):
            with metrics.aggregate("train_inner") as agg:
                metrics.log_scalar("loss2", j / 10.0)

print(metrics.get_smoothed_values("train_inner")["loss2"])

#
# with metrics.aggregate("train"):
#     for step, batch in enumerate(epoch):
#         with metrics.aggregate("train_inner") as agg:
#             metrics.log_scalar("loss", get_loss(batch))
#             if step % log_interval == 0:
#                 print(agg.get_smoothed_value("loss"))
#                 agg.reset()
# print(metrics.get_smoothed_values("train")["loss"])
