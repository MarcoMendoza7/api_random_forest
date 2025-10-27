from rest_framework import serializers

class ResultSerializer(serializers.Serializer):
    f1_scaled = serializers.FloatField()
    f1_unscaled = serializers.FloatField()
    comparison = serializers.CharField()
