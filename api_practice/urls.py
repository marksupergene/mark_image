from django.urls import include, path
from rest_framework import routers
import api_practice.api as api
import api_practice.api2 as api2

router = routers.DefaultRouter()

urlpatterns = [
    path('', include(router.urls)),
    path('ig_filter', api.ig_filter),
    path('old_filter', api.old_filter),
    path('mona_lisa', api2.mona_lisa),
    path('friends_morph', api2.friends_morph),
]
