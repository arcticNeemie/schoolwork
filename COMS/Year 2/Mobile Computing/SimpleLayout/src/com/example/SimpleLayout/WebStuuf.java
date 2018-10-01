package com.example.SimpleLayout;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.webkit.WebView;
import android.webkit.WebViewClient;

/**
 * Created by 1391758 on 2017/05/17.
 */
public class WebStuuf extends Activity {
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.d("INTENT","Next activity started");
        Intent intent = getIntent();
        String value = intent.getStringExtra("key");
        //Uri address = Uri.parse(value);
        super.onCreate(savedInstanceState);
        setContentView(R.layout.web);

        WebView wv = (WebView) findViewById(R.id.webView);

        //wv.getSettings().setJavaScriptEnabled(true);
        Log.d("WEB", "URL = " + value);
        wv.setWebViewClient(new WebViewClient());
        //wv.setClickable(false);
        wv.loadUrl(value);

    }
}