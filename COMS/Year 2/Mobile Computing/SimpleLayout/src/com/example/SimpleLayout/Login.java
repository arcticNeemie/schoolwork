package com.example.SimpleLayout;

import android.app.ActionBar;
import android.app.Activity;
import android.app.ProgressDialog;
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
import android.view.inputmethod.InputMethodManager;
import android.widget.*;
import org.json.JSONArray;
import org.json.JSONObject;
import org.w3c.dom.Text;

public class Login extends Activity {

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.login_menu, menu);
        ActionBar actionBar = getActionBar();
        actionBar.setDisplayShowTitleEnabled(true);
        actionBar.show();
        return true;
    }    /**
     * Called when the activity is first created.
     */
    TextView register;
    Button login;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.login);
        Context me = this;
        register = (TextView)findViewById(R.id.registerText);
        login = (Button)findViewById(R.id.loginButton);
        register.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent goRegister = new Intent(me, Register.class);
                startActivity(goRegister);
                //Log.d("INTENT", "switching over...")
            }
        });
        login.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                try {
                    InputMethodManager inputManager = (InputMethodManager)
                            getSystemService(Context.INPUT_METHOD_SERVICE);

                    inputManager.hideSoftInputFromWindow(getCurrentFocus().getWindowToken(),
                            InputMethodManager.HIDE_NOT_ALWAYS);
                }catch (Exception e)
                {

                }

                TextView registerText = (TextView) findViewById(R.id.registerText);


                TextView remind = (TextView) findViewById(R.id.reminderText);
                remind.setText("Logging in ...");

                EditText email = (EditText) findViewById(R.id.emailEdit);
                String user_email = email.getText().toString();

                EditText password = (EditText) findViewById(R.id.passwdEdit);
                String user_password = password.getText().toString();

                ContentValues params = new ContentValues();
                params.put("email",user_email);
                params.put("password",user_password);
                AsyncHTTPRequest AsyncLogin = new AsyncHTTPRequest(
                        "http://lamp.ms.wits.ac.za/~s1312548/login.php",params) {
                    @Override
                    protected void onPostExecute(String output) {
                        if (output.equals("mellon"))
                        {
                            Intent intent = new Intent(Login.this, main.class);
                            intent.putExtra("email", user_email);
                            Login.this.startActivity(intent);
                        }
                        else
                        {
                            remind.setText("Username or Password is incorrect");
                            Log.d("LOGIN", output);
                        }

                    }
                };
                AsyncLogin.execute();

                /*AsyncHTTPRequest refresh_stuff = new AsyncHTTPRequest(
                        "http://lamp.ms.wits.ac.za/~s1312548/update_articles.php",params) {
                    @Override
                    protected void onPostExecute(String output) {
                    }
                };
                refresh_stuff.execute();*/
            }
        });
    }


    public void get_help(MenuItem item) {
        String url = "http://lamp.ms.wits.ac.za/~s1312548/help.html";
        Intent myIntent = new Intent(Login.this, WebStuuf.class);
        myIntent.putExtra("key", url);
        Login.this.startActivity(myIntent);
    }
}
