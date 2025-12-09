# Burn YOLOX test

Model and script taken from [here](https://github.com/tracel-ai/models).

Powered by :
<a href="https://github.com/anuraghazra/convoychat">
<img height=50 align="center" src="https://go-skill-icons.vercel.app/api/icons?i=burn" />
</a>


**Bounding boxes result :**

<img src="samples/dog_bike_man.output.png" alt="drawing" style="width:500px;"/>


## Former issue (now fixed)
Used corrected source : 
```code
diff --git a/yolox-burn/src/model/bottleneck.rs b/yolox-burn/src/model/bottleneck.rs
index 1b738cd..b6f8d18 100644
--- a/yolox-burn/src/model/bottleneck.rs
+++ b/yolox-burn/src/model/bottleneck.rs
@@ -111,6 +111,7 @@ impl SppBottleneckConfig {
             .map(|k| {
                 let pad = k / 2;
                 MaxPool2dConfig::new([k, k])
+                    .with_strides([1, 1])
                     .with_padding(burn::nn::PaddingConfig2d::Explicit(pad, pad))
             })
             .collect();

```


