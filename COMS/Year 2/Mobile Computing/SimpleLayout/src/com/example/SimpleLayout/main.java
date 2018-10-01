package com.example.SimpleLayout;

import android.app.ActionBar;
import android.app.Activity;
import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
import android.graphics.Color;
import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.text.Html;
import android.util.Log;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.InputStream;
import java.net.URL;

public class main extends Activity {
    /**
     * Called when the activity is first created.
     */
    RSSParser feed;

    @Override
    public void onBackPressed() {
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.main_menu, menu);
        ActionBar actionBar = getActionBar();
        actionBar.setDisplayShowTitleEnabled(true);
        actionBar.show();
        return true;
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Context ctx = this;



        ActionBar actionBar = getActionBar();
        actionBar.setDisplayShowTitleEnabled(true);
        actionBar.setTitle("Today's News");
        actionBar.setSubtitle("tailored especially for you");
        actionBar.show();

        /*AsyncHTTPRequest asyncHttpRequest = new AsyncHTTPRequest(
                "rss.nytimes.com/services/xml/rss/nyt/World.xml",params) {
            @Override
            protected void onPostExecute(String output) {
                feed = new RSSParser(output);
                ArrayList<String> al = new ArrayList<String>();
                al = feed.titles;

                LinearLayout big_layout = (LinearLayout) main.this.findViewById(R.id.BigLinearLayout);
                for (String s : al)
                {
                    LinearLayout vertical_outer_layout = new LinearLayout(main.this);
                    vertical_outer_layout.setLayoutParams(new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT));
                    vertical_outer_layout.setOrientation(LinearLayout.VERTICAL);

                    big_layout.addView(vertical_outer_layout);

                    LinearLayout horizontal_inner_layout = new LinearLayout(main.this);
                    horizontal_inner_layout.setLayoutParams(new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT));
                    horizontal_inner_layout.setOrientation(LinearLayout.HORIZONTAL);
                    vertical_outer_layout.addView(horizontal_inner_layout);

                    ImageView article_image = new ImageView(main.this);
                    article_image.setMaxHeight(128);
                    article_image.setMaxWidth(128);
                    article_image.setAdjustViewBounds(true);

                    Drawable d = LoadImageFromWebOperations(feed.getImageForTitle(s));
                    if ( d != null)
                    {
                        article_image.setImageDrawable(d);
                    }
                    else
                    {
                        article_image.setImageResource(R.drawable.ic_newspaper_black_48dp);
                    }

                    horizontal_inner_layout.addView(article_image);

                    LinearLayout vertical_inner_layout = new LinearLayout(main.this);
                    vertical_inner_layout.setLayoutParams(new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT));
                    vertical_inner_layout.setOrientation(LinearLayout.VERTICAL);
                    vertical_outer_layout.addView(vertical_inner_layout);
                    vertical_outer_layout.setOnClickListener(new View.OnClickListener() {
                        @Override
                        public void onClick(View view) {
                            String url = feed.getLinkForTitle(s);
                            Intent myIntent = new Intent(main.this, WebStuuf.class);
                            myIntent.putExtra("key", url);
                            main.this.startActivity(myIntent);
                        }
                    });

                    TextView title_text_view = new TextView(main.this);
                    title_text_view.setTextSize(18);
                    title_text_view.setText(s);
                    title_text_view.setTextColor(Color.BLACK);


                    horizontal_inner_layout.addView(title_text_view);
                    TextView description_text_view = new TextView(main.this);
                    description_text_view.setText(feed.getTextForTitle(s));
                    title_text_view.setMaxLines(2);
                    vertical_inner_layout.addView(description_text_view);
                }
        }
        };
        asyncHttpRequest.execute();*/


        setContentView(R.layout.main);
        update();
    }


    public Drawable LoadImageFromWebOperations(String url) {
        try {
            InputStream is = (InputStream) new URL(url).getContent();
            Drawable d = Drawable.createFromStream(is, "src name");
            return d;
        } catch (Exception e) {
            return null;
        }
    }

    public void processJSON(String json){
        try {
            JSONArray all = new JSONArray(json);
            for (int i=0; i<all.length(); i++){
                JSONObject item=all.getJSONObject(i);
                String title = item.getString("title");
                String description = item.getString("description");
                System.out.println(title + " - "+description);
            }
        } catch (JSONException e) {
            e.printStackTrace();
        }
    }

    public void update()
    {
        ContentValues params = new ContentValues();
        String user_email = getIntent().getStringExtra("email");
        params.put("email",user_email);
        AsyncHTTPRequest asyncHttpRequest = new AsyncHTTPRequest(
                "http://lamp.ms.wits.ac.za/~s1312548/get_user_articles.php",params) {
            @Override
            protected void onPostExecute(String output) {
                try {
                    JSONArray articles = new JSONArray(output);
                    LinearLayout big_layout = (LinearLayout) main.this.findViewById(R.id.BigLinearLayout);
                    big_layout.removeAllViews();

                    if (articles.length() == 0)
                    {
                        TextView whoops = new TextView(main.this);
                        whoops.setText("It looks like you aren't subscribed to anything");
                        big_layout.addView(whoops);
                    }

                    for (int j=0; j<articles.length(); j++)
                    {
                        JSONObject item=articles.getJSONObject(j);
                        String title = item.getString("ARTICLE_TITLE");
                        String description = item.getString("ARTICLE_DESCRIPTION");
                        String link = item.getString("ARTICLE_URL");

                        Log.d("title", title);
                        Log.d("desc", description);
                        Log.d("link", link);


                        LinearLayout vertical_outer_layout = new LinearLayout(main.this);
                        vertical_outer_layout.setLayoutParams(new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT));
                        vertical_outer_layout.setOrientation(LinearLayout.VERTICAL);

                        big_layout.addView(vertical_outer_layout);

                        LinearLayout horizontal_inner_layout = new LinearLayout(main.this);
                        horizontal_inner_layout.setLayoutParams(new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT));
                        horizontal_inner_layout.setOrientation(LinearLayout.HORIZONTAL);
                        vertical_outer_layout.addView(horizontal_inner_layout);

                        ImageView article_image = new ImageView(main.this);
                        article_image.setMaxHeight(128);
                        article_image.setMaxWidth(128);
                        article_image.setAdjustViewBounds(true);

                        Drawable d = null;
                        if ( d != null)
                        {
                            article_image.setImageDrawable(d);
                        }
                        else
                        {
                            article_image.setImageResource(R.drawable.ic_newspaper_black_48dp);
                        }

                        horizontal_inner_layout.addView(article_image);

                        LinearLayout vertical_inner_layout = new LinearLayout(main.this);
                        vertical_inner_layout.setLayoutParams(new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT));
                        vertical_inner_layout.setOrientation(LinearLayout.VERTICAL);
                        vertical_outer_layout.addView(vertical_inner_layout);
                        vertical_outer_layout.setOnClickListener(new View.OnClickListener() {
                            @Override
                            public void onClick(View view) {
                                String url = link;
                                Intent myIntent = new Intent(main.this, WebStuuf.class);
                                myIntent.putExtra("key", url);
                                main.this.startActivity(myIntent);
                            }
                        });

                        TextView title_text_view = new TextView(main.this);
                        title_text_view.setTextSize(18);
                        title_text_view.setText(title);
                        title_text_view.setTextColor(Color.BLACK);


                        horizontal_inner_layout.addView(title_text_view);
                        TextView description_text_view = new TextView(main.this);
                        description_text_view.setText(Html.fromHtml(description));
                        description_text_view.setMaxLines(3);
                        title_text_view.setMaxLines(2);
                        vertical_inner_layout.addView(description_text_view);

                    }


                }
                catch (Exception e)
                {

                }
            }
        };
        asyncHttpRequest.execute();

    }

    public void update(MenuItem item) {
        update();
    }

    public void addTags(MenuItem item) {
        Intent openTags = new Intent(main.this,AddTags.class);
        String user_email = getIntent().getStringExtra("email");
        openTags.putExtra("email",user_email);
        startActivity(openTags);
        update();
    }

    public void get_help(MenuItem item) {
        String url = "http://lamp.ms.wits.ac.za/~s1312548/help.html";
        Intent myIntent = new Intent(main.this, WebStuuf.class);
        myIntent.putExtra("key", url);
        main.this.startActivity(myIntent);
        update();
    }

    public void addFeeds(MenuItem item) {
        Intent openFeeds = new Intent(main.this,AddFeeds.class);
        String user_email = getIntent().getStringExtra("email");
        openFeeds.putExtra("email",user_email);
        startActivity(openFeeds);
        update();
    }
}
