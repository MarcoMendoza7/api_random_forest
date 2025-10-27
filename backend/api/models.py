from django.db import models

class TrainingSession(models.Model):
    dataset_percentage = models.FloatField()
    selected_tree = models.IntegerField()
    f1_scaled = models.FloatField()
    f1_unscaled = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Session {self.id} - {self.dataset_percentage * 100:.0f}% data"
