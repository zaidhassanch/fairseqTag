

import metrics

with metrics.aggregate("train"):
    for i in range(10):
        print(i)
        metrics.log_scalar("loss", i/10.0)
print(metrics.get_smoothed_values("train")["loss"])

#
# with metrics.aggregate("train"):
#     for step, batch in enumerate(epoch):
#         with metrics.aggregate("train_inner") as agg:
#             metrics.log_scalar("loss", get_loss(batch))
#             if step % log_interval == 0:
#                 print(agg.get_smoothed_value("loss"))
#                 agg.reset()
# print(metrics.get_smoothed_values("train")["loss"])
